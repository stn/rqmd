//! ANSI palette honouring `NO_COLOR` and `--no-color` plus terminal detection.
//!
//! Maps to qmd's `c.*` constants in `src/cli/qmd.ts` (lines 196–206).

use std::io::IsTerminal;

pub struct Palette {
    pub enabled: bool,
}

impl Palette {
    pub fn new(force_off: bool) -> Self {
        let enabled =
            !force_off && std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal();
        Self { enabled }
    }

    fn code(&self, s: &'static str) -> &'static str {
        if self.enabled {
            s
        } else {
            ""
        }
    }

    pub fn reset(&self) -> &'static str {
        self.code("\x1b[0m")
    }
    pub fn dim(&self) -> &'static str {
        self.code("\x1b[2m")
    }
    pub fn bold(&self) -> &'static str {
        self.code("\x1b[1m")
    }
    pub fn cyan(&self) -> &'static str {
        self.code("\x1b[36m")
    }
    pub fn yellow(&self) -> &'static str {
        self.code("\x1b[33m")
    }
    pub fn green(&self) -> &'static str {
        self.code("\x1b[32m")
    }
}
