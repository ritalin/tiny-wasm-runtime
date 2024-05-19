use crate::binary::section::SectionCode;
use nom::{bytes::complete::{tag, take}, multi::many0, number::complete::{le_u32, le_u8}, sequence::pair, IResult};
use nom_leb128::leb128_u32;
use num_traits::FromPrimitive;

use super::{instruction::Instruction, opcode::Opcode, section::{Function, FunctionLocal}, types::{FuncType, ValueType}};

const WASM_MAGIC: &str = "\0asm";

#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    pub magic: String,
    pub version: u32,
    pub type_section: Option<Vec<FuncType>>,
    pub fn_section: Option<Vec<u32>>,
    pub code_section: Option<Vec<Function>>,
}

impl Default for Module {
    fn default() -> Self {
        Self { magic: WASM_MAGIC.to_string(), version: 1, type_section: None, fn_section: None, code_section: None }
    }
}

impl Module {
    pub fn new(wasm: &[u8]) -> anyhow::Result<Self> {
        let (_, module) = Module::decode(wasm)
            .map_err(|err| anyhow::anyhow!("failed to parse wasm: {}", err))?;
        Ok(module)
    }

    fn decode(input: &[u8]) -> IResult<&[u8], Module> {
        let (input, _) = tag(WASM_MAGIC.as_bytes())(input)?;
        let (input, version) = le_u32(input)?;

        let mut module = Module { magic: WASM_MAGIC.to_string(), version, ..Default::default() };

        let mut remaining = input;
        
        while !remaining.is_empty() {
            match decode_section_header(remaining) {
                Ok((input, (code, sz))) => {
                    let (rest, section_contents) = take(sz)(input)?;

                    match code {
                        SectionCode::Type => {
                            let (_, tys) = decode_type_section(section_contents)?;
                            module.type_section = Some(tys);
                        }
                        SectionCode::Function => {
                            let (_, fns) = decode_function_section(section_contents)?;
                            module.fn_section = Some(fns);
                        }
                        SectionCode::Code => {
                            let (_, code) = decode_code_section(section_contents)?;
                            module.code_section = Some(code);
                        }
                        _ => unreachable!()
                    }
                    remaining = rest;
                }
                Err(err) => return Err(err),
            }
        }

        Ok((input, module))
    }
}

fn decode_section_header(input: &[u8]) -> IResult<&[u8], (SectionCode, u32)> {
    let (input, (code, sz)) = pair(le_u8, leb128_u32)(input)?;
    Ok((
        input, 
        (
            SectionCode::from_u8(code).expect(&format!("Unexpected section code: {code}")),
            sz,
        )
    ))
}

fn decodea_value_type(input: &[u8]) -> IResult<&[u8], ValueType> {
    let (input, v) = le_u8(input)?;
    Ok((input, v.into()))
}

fn decode_type_section(input: &[u8]) -> IResult<&[u8], Vec<FuncType>> {
    let (mut input, type_count) = leb128_u32(input)?;
    let mut fns = vec![];

    for _ in 0..type_count {
        let(rest, _) = le_u8(input)?; // omit fn sig
        // decode fn parameter types
        let(rest, count) = leb128_u32(rest)?;
        let(rest, tys) = take(count)(rest)?;
        let(_, params) = many0(decodea_value_type)(tys)?;
        
        // decode fn return types
        let(rest, count) = leb128_u32(rest)?;
        let(rest, tys) = take(count)(rest)?;
        let(_, returns) = many0(decodea_value_type)(tys)?;

        fns.push(FuncType { params, returns });
        input = rest;
    }

    Ok((input, fns))
}

fn decode_function_section(input: &[u8]) -> IResult<&[u8], Vec<u32>> {
    let mut fns = vec![];
    let (mut input, count) = leb128_u32(input)?;
    
    for _ in 0..count {
        let (rest, i) = leb128_u32(input)?;
        fns.push(i);
        input = rest;
    }
    
    Ok((input, fns))
}

fn decode_opcode(input: &[u8]) -> IResult<&[u8], Opcode> {
    let (input, opcode) = le_u8(input)?;

    use nom::error::Error as NomError;
    use nom::error::ErrorKind as NomErrKind;
    Opcode::from_u8(opcode).map(|x| (input, x)).ok_or_else(|| nom::Err::Failure(NomError { input, code: NomErrKind::Verify }))
}

fn decode_instruction(input: &[u8]) -> IResult<&[u8], Instruction> {
    let (input, op) = decode_opcode(input)?;

    match op {
        Opcode::End => Ok((input, Instruction::End)),
        Opcode::LocalGet => {
            let (input, i) = leb128_u32(input)?;
            Ok((input, Instruction::LocalGet(i)))
        }
        Opcode::I32Add => Ok((input, Instruction::I32Add)),
    }
}

fn decode_function_body(input: &[u8]) -> IResult<&[u8], Function> {
    let mut locals = vec![];

    // decode local vars
    let (mut input, count) = leb128_u32(input)?;

    for _ in 0..count {
        let (rest, type_count) = leb128_u32(input)?;
        let (rest, value_type) = decodea_value_type(rest)?;

        locals.push(FunctionLocal { type_count, value_type });

        input = rest;
    }

    // decode instructions
    let mut code = vec![];
    let mut remaining = input;

    while !remaining.is_empty() {
        let (rest, inst) = decode_instruction(remaining)?;
        code.push(inst);

        remaining = rest;
    }

    Ok((input, Function {
        locals,
        code,
    }))
}

fn decode_code_section(input: &[u8]) -> IResult<&[u8], Vec<Function>> {
    let mut fns = vec![];
    let (mut input, count) = leb128_u32(input)?;

    for _ in 0..count {
        let (rest, sz_body) = leb128_u32(input)?;
        let (rest, contents_body) = take(sz_body)(rest)?;
        let (_, bodies) = decode_function_body(contents_body)?;

        fns.push(bodies);

        input = rest;
    }

    Ok((input, fns))
}

#[cfg(test)]
mod decoder_tests {
    use crate::binary::{instruction::Instruction, module::Module, section::{Function, FunctionLocal, SectionCode}, types::{FuncType, ValueType}};
    use anyhow::Result;

    #[test]
    fn decode_simplest_module() -> Result<()> {
        // プリアンブルのみのwasm
        let wasm = wat::parse_str("(module)")?;
        let module = Module::new(&wasm)?;
        assert_eq!(module, Module::default());
        Ok(())
    }

    #[test]
    fn decode_simplest_fn() -> Result<()> {
        let wasm = wat::parse_str("(module (func))")?;
        let module = Module::new(&wasm)?;

        let expected = Module {
            type_section: Some(vec![FuncType::default()]),
            fn_section: Some(vec![0]),
            code_section: Some(vec![Function { locals: vec![], code: vec![Instruction::End] }]),
            ..Default::default()
        };

        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_simplest_fn_with_args() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i64)))")?;
        let module = Module::new(&wasm)?;
        let expected = Module {
            type_section: Some(vec![
                FuncType { 
                    params: vec![ValueType::I32, ValueType::I64], 
                    returns: vec![] }
            ]),
            fn_section: Some(vec![0]),
            code_section: Some(vec![Function { locals: vec![], code: vec![Instruction::End] }]),
            ..Default::default()
        };

        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_simplest_fn_local_vars() -> Result<()> {
        let wasm = wat::parse_str("(module (func (local i32) (local i64 i64)))")?;
        let module = Module::new(&wasm)?;
        let expected = Module {
            type_section: Some(vec![FuncType::default()]),
            fn_section: Some(vec![0]),
            code_section: Some(vec![Function {
                locals: vec![
                    FunctionLocal{ type_count:1, value_type: ValueType::I32 },
                    FunctionLocal{ type_count: 2, value_type: ValueType::I64 }, 
                ], 
                code: vec![Instruction::End] }]),
            ..Module::default()
        };

        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_simplest_fn_with_returns() -> Result<()> {
        let wasm = wat::parse_str("(module (func (result i32)))")?;
        let module = Module::new(&wasm)?;
        let expected = Module {
            type_section: Some(vec![
                FuncType { 
                    params: vec![], 
                    returns: vec![ValueType::I32] }
            ]),
            fn_section: Some(vec![0]),
            code_section: Some(vec![Function { locals: vec![], code: vec![Instruction::End] }]),
            ..Default::default()
        };
        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_fn_add() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let module = Module::new(&wasm)?;
        let expected = Module {
            type_section: Some(vec![
                FuncType { 
                    params: vec![ValueType::I32, ValueType::I32], 
                    returns: vec![ValueType::I32] }
            ]),
            fn_section: Some(vec![0]),
            code_section: Some(vec![Function { locals: vec![], code: vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32Add,
                Instruction::End
            ] }]),
            ..Default::default()
        };
        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_section_headers() -> Result<()> {
        assert_eq!((SectionCode::Type, 4u32), super::decode_section_header(&[0x01, 0x04])?.1);
        assert_eq!((SectionCode::Function, 2u32), super::decode_section_header(&[0x03, 0x02])?.1);
        assert_eq!((SectionCode::Code, 4u32), super::decode_section_header(&[0x0a, 0x04])?.1);
        Ok(())
    }

    #[test]
    fn decodea_value_types() -> Result<()> {
        assert_eq!(ValueType::I32, super::decodea_value_type(&[0x7F])?.1);
        assert_eq!(ValueType::I64, super::decodea_value_type(&[0x7E])?.1);
        Ok(())
    }

    #[test]
    fn decode_type_sections() -> Result<()> {
        let ret = super::decode_type_section(&[0x01, 0x60, 0x02, 0x7F, 0x7E, 0])?.1;
        assert_eq!(1, ret.len());
        assert_eq!(vec![ValueType::I32, ValueType::I64], ret[0].params);
        assert_eq!(Vec::<ValueType>::new(), ret[0].returns);
        Ok(())
    }

    #[test]
    fn decode_function_sections() -> Result<()> {
        let ret = super::decode_function_section(&[0x03, 0, 0x01, 0, 0x0a])?.1;
        assert_eq!(3, ret.len());
        assert_eq!(vec![0, 1, 0], ret);
        Ok(())
    }

    #[test]
    fn decode_instructions() -> Result<()> {
        assert_eq!(Instruction::End, super::decode_instruction(&[0x0B])?.1);
        assert_eq!(Instruction::I32Add, super::decode_instruction(&[0x6A])?.1);
        assert_eq!(Instruction::LocalGet(1), super::decode_instruction(&[0x20, 1])?.1);
        Ok(())
    }
}
