use num_derive::FromPrimitive;

use super::{instruction::Instruction, types::ValueType};

#[derive(Debug, PartialEq, Eq, FromPrimitive)]
pub enum SectionCode {
    Type = 0x01,
    Function = 0x03,
    Export = 0x07,
    Import = 0x02,
    Code = 0x0a,
    Custom = 0,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionLocal {
    pub type_count: u32,
    pub value_type: ValueType,
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct Function {
    pub locals: Vec<FunctionLocal>,
    pub code: Vec<Instruction>,
}