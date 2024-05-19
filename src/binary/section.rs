use num_derive::FromPrimitive;

use super::instruction::Instruction;

#[derive(Debug, PartialEq, Eq, FromPrimitive)]
pub enum SectionCode {
    Type = 0x01,
    Function = 0x03,
    Code = 0x0a,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionLocal {

}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct Function {
    pub locals: Vec<FunctionLocal>,
    pub code: Vec<Instruction>,
}