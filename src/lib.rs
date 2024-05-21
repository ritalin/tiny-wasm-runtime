use anyhow::Result;
use clap::Parser;

pub mod binary;
pub mod execution;

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct CmdConfig {
    #[arg(value_name="FILE", help="wat-format file path", default_value="-", required=true)]
    pub file: String,
}

pub fn get_args() -> Result<CmdConfig> {
    Ok(CmdConfig::parse())
}