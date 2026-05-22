//! The MCP [`ServerHandler`] implementation shared by stdio and HTTP transports.
//!
//! Port of `createMcpServer` in `tobi/qmd/src/mcp/server.ts:171-538`: registers
//! the `qmd://{+path}` resource template, the four tools, and the dynamic
//! `instructions`. Tool names match qmd; the server identity and user-facing
//! wording are rqmd-ified (see [`crate::tools`]).
//!
//! Store access goes through [`crate::worker::StoreHandle`] (see that module for
//! why the `!Sync` store is confined to its own thread).

use rmcp::ErrorData as McpError;
use rmcp::ServerHandler;
use rmcp::model::{
    AnnotateAble, CallToolRequestParams, CallToolResult, ListResourceTemplatesResult,
    ListToolsResult, PaginatedRequestParams, RawResourceTemplate, ReadResourceRequestParams,
    ReadResourceResult, ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
};
use rmcp::service::{RequestContext, RoleServer};
use serde_json::Value;

use rqmd_core::RqmdStore;

use crate::tools;
use crate::worker::{self, StoreHandle};

/// Shared MCP server state. Cheap to clone (handle + cached instructions),
/// which the HTTP transport's per-session factory needs.
#[derive(Clone)]
pub struct QmdMcpServer {
    handle: StoreHandle,
    /// Precomputed once at server creation (qmd builds instructions once in
    /// `createMcpServer`), so `get_info` — which is synchronous — can return it
    /// directly without touching the store worker.
    instructions: Option<String>,
}

impl QmdMcpServer {
    /// Build the server from an owned store: computes the instructions once,
    /// then moves the store onto its dedicated worker thread.
    pub fn new(store: RqmdStore) -> Self {
        let instructions = tools::build_instructions(&store);
        let handle = worker::spawn(store);
        Self {
            handle,
            instructions,
        }
    }

    /// Clone of the store handle (for the HTTP REST endpoints).
    pub fn handle(&self) -> StoreHandle {
        self.handle.clone()
    }

    /// Tear down LLM workers. Call once after the transport finishes.
    pub async fn shutdown(&self) {
        self.handle.shutdown().await;
    }

    /// The four tool definitions (names, titles, descriptions, schemas, and the
    /// `readOnlyHint: true, openWorldHint: false` annotations from qmd).
    fn tool_defs() -> Vec<Tool> {
        let read_only = || ToolAnnotations::new().read_only(true).open_world(false);
        vec![
            Tool::new(
                tools::TOOL_QUERY,
                tools::QUERY_DESCRIPTION,
                tools::query_input_schema(),
            )
            .with_title("Query")
            .annotate(read_only()),
            Tool::new(
                tools::TOOL_GET,
                tools::GET_DESCRIPTION,
                tools::get_input_schema(),
            )
            .with_title("Get Document")
            .annotate(read_only()),
            Tool::new(
                tools::TOOL_MULTI_GET,
                tools::MULTI_GET_DESCRIPTION,
                tools::multi_get_input_schema(),
            )
            .with_title("Multi-Get Documents")
            .annotate(read_only()),
            Tool::new(
                tools::TOOL_STATUS,
                tools::STATUS_DESCRIPTION,
                tools::status_input_schema(),
            )
            .with_title("Index Status")
            .annotate(read_only()),
        ]
    }
}

impl ServerHandler for QmdMcpServer {
    fn get_info(&self) -> ServerInfo {
        // `ServerInfo` is `#[non_exhaustive]`; build from Default and mutate.
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_resources()
            .build();
        // Deliberate divergence from server.ts's "qmd" (user-requested): the
        // server identifies as `rqmd`. Clients namespace tools by this name.
        info.server_info.name = "rqmd".to_string();
        info.server_info.version = env!("CARGO_PKG_VERSION").to_string();
        info.instructions = self.instructions.clone();
        // protocol_version stays the rmcp default (LATEST); rmcp negotiates down
        // to the client's requested version during initialize.
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(Self::tool_defs()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let args = Value::Object(request.arguments.unwrap_or_default());
        self.handle.call_tool(request.name.to_string(), args).await
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        // `qmd://{+path}` — `{+path}` is RFC 6570 reserved expansion (slashes
        // preserved). No `list` (documents are discovered via search).
        let template = RawResourceTemplate::new("qmd://{+path}", "document")
            .with_title("RQMD Document")
            .with_description(
                "A markdown document from your rqmd knowledge base. Use search tools to discover documents.",
            )
            .with_mime_type("text/markdown")
            .no_annotation();
        Ok(ListResourceTemplatesResult::with_all_items(vec![template]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let uri = request.uri;
        let path_part = uri.strip_prefix("qmd://").unwrap_or(uri.as_str());
        let decoded = percent_decode(path_part);
        self.handle.read_resource(decoded, uri).await
    }
}

/// Decode `%XX` escapes like JS `decodeURIComponent` (used on `qmd://` resource
/// URIs). `+` is left as-is (decodeURIComponent does not treat it as a space).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2]))
        {
            out.push(h * 16 + l);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_defs_cover_all_four() {
        let names: Vec<String> = QmdMcpServer::tool_defs()
            .iter()
            .map(|t| t.name.to_string())
            .collect();
        assert_eq!(names, vec!["query", "get", "multi_get", "status"]);
    }

    #[test]
    fn tools_are_read_only_closed_world() {
        for t in QmdMcpServer::tool_defs() {
            let a = t.annotations.expect("annotations present");
            assert_eq!(a.read_only_hint, Some(true));
            assert_eq!(a.open_world_hint, Some(false));
        }
    }

    #[test]
    fn percent_decode_handles_slashes_spaces_and_colons() {
        assert_eq!(percent_decode("readme.md"), "readme.md");
        assert_eq!(
            percent_decode("meetings%2Fmeeting-2024-01.md"),
            "meetings/meeting-2024-01.md"
        );
        assert_eq!(
            percent_decode("External%20Podcast%2F2023%20April%20-%20Interview.md"),
            "External Podcast/2023 April - Interview.md"
        );
        assert_eq!(percent_decode("api.md%3A10"), "api.md:10");
        // Lone % or truncated escape passes through.
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("a%2"), "a%2");
        // '+' is not a space.
        assert_eq!(percent_decode("a+b"), "a+b");
    }
}
