use anyhow::Result;
use clap::Parser;

pub mod binary;
pub mod execution;

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct CmdConfig {
    #[arg(value_name="FILE", help="wat-format file path", default_value="-", required=true)]
    pub file: String,
    #[arg(long, help="passing numeric value(s)", value_parser=clap::value_parser!(i32))]
    pub args: Vec<i32>,
    #[arg(long)]
    pub disable_ansi_color: bool,
}

pub fn get_args() -> Result<CmdConfig> {
    Ok(CmdConfig::parse())
}