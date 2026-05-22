//! Store actor: confines the `!Sync` [`RqmdStore`] to a dedicated thread.
//!
//! `RqmdStore` wraps a `rusqlite::Connection`, which is `Send` but not `Sync`,
//! so a shared `&RqmdStore` cannot be held across an `.await` in a `Send`
//! future — yet rmcp's (non-`local`) `ServerHandler` requires `Send` futures,
//! and the Streamable HTTP service is unavailable under the `local` feature.
//!
//! The fix: move the store onto its own thread running a current-thread tokio
//! runtime (where futures need not be `Send`) and talk to it over channels. The
//! [`StoreHandle`] holds only a `Send + Sync` `mpsc::Sender`, so every MCP
//! handler future stays `Send`. All requests serialize through this one worker,
//! which also matches the single-connection / single-user nature of the store.

use rmcp::ErrorData as McpError;
use rmcp::model::{CallToolResult, ReadResourceResult};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use rqmd_core::RqmdStore;

use crate::tools::{self, QueryArgs, SearchResultItem};

const CHANNEL_CAPACITY: usize = 32;

enum StoreCommand {
    CallTool {
        name: String,
        args: Value,
        reply: oneshot::Sender<Result<CallToolResult, McpError>>,
    },
    ReadResource {
        decoded: String,
        uri: String,
        reply: oneshot::Sender<Result<ReadResourceResult, McpError>>,
    },
    RunQuery {
        args: QueryArgs,
        reply: oneshot::Sender<Result<Vec<SearchResultItem>, McpError>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

/// Cheap-to-clone handle to the store worker. Clones share the one worker.
#[derive(Clone)]
pub struct StoreHandle {
    tx: mpsc::Sender<StoreCommand>,
}

/// Move `store` onto a dedicated worker thread and return a handle to it.
pub fn spawn(store: RqmdStore) -> StoreHandle {
    let (tx, mut rx) = mpsc::channel::<StoreCommand>(CHANNEL_CAPACITY);
    std::thread::Builder::new()
        .name("rqmd-mcp-store".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build current-thread runtime for rqmd-mcp store worker");
            rt.block_on(async move {
                while let Some(cmd) = rx.recv().await {
                    match cmd {
                        StoreCommand::CallTool { name, args, reply } => {
                            let _ = reply.send(dispatch_tool(&store, &name, args).await);
                        }
                        StoreCommand::ReadResource {
                            decoded,
                            uri,
                            reply,
                        } => {
                            let _ = reply.send(tools::handle_read_resource(&store, &decoded, &uri));
                        }
                        StoreCommand::RunQuery { args, reply } => {
                            let _ = reply.send(tools::run_query(&store, &args).await);
                        }
                        StoreCommand::Shutdown { reply } => {
                            store.shutdown().await;
                            let _ = reply.send(());
                            return;
                        }
                    }
                }
                // All handles dropped: dispose LLM workers before the thread ends.
                store.shutdown().await;
            });
        })
        .expect("spawn rqmd-mcp store worker thread");
    StoreHandle { tx }
}

async fn dispatch_tool(
    store: &RqmdStore,
    name: &str,
    args: Value,
) -> Result<CallToolResult, McpError> {
    match name {
        tools::TOOL_QUERY => tools::handle_query(store, args).await,
        tools::TOOL_GET => tools::handle_get(store, args).await,
        tools::TOOL_MULTI_GET => tools::handle_multi_get(store, args).await,
        tools::TOOL_STATUS => tools::handle_status(store).await,
        other => Err(McpError::invalid_params(
            format!("unknown tool: {other}"),
            None,
        )),
    }
}

fn worker_gone() -> McpError {
    McpError::internal_error("rqmd store worker is unavailable", None)
}

impl StoreHandle {
    pub async fn call_tool(&self, name: String, args: Value) -> Result<CallToolResult, McpError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(StoreCommand::CallTool { name, args, reply })
            .await
            .map_err(|_| worker_gone())?;
        rx.await.map_err(|_| worker_gone())?
    }

    pub async fn read_resource(
        &self,
        decoded: String,
        uri: String,
    ) -> Result<ReadResourceResult, McpError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(StoreCommand::ReadResource {
                decoded,
                uri,
                reply,
            })
            .await
            .map_err(|_| worker_gone())?;
        rx.await.map_err(|_| worker_gone())?
    }

    pub async fn run_query(&self, args: QueryArgs) -> Result<Vec<SearchResultItem>, McpError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(StoreCommand::RunQuery { args, reply })
            .await
            .map_err(|_| worker_gone())?;
        rx.await.map_err(|_| worker_gone())?
    }

    /// Dispose LLM workers. Best-effort: if the worker already exited, this is a
    /// no-op.
    pub async fn shutdown(&self) {
        let (reply, rx) = oneshot::channel();
        if self
            .tx
            .send(StoreCommand::Shutdown { reply })
            .await
            .is_ok()
        {
            let _ = rx.await;
        }
    }
}
