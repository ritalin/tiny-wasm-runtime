use crate::binary::section::SectionCode;
use nom::{bytes::complete::{tag, take}, multi::many0, number::complete::{le_u32, le_u8}, sequence::pair, IResult};
use nom_leb128::{leb128_i32, leb128_u32};
use num_traits::FromPrimitive;

use super::{instruction::Instruction, opcode::Opcode, section::{Function, FunctionLocal}, types::{Data, Export, ExportDesc, FuncType, Import, ImportDesc, Memory, ValueType}};

const WASM_MAGIC: &str = "\0asm";

#[derive(Debug, PartialEq, Eq)]
pub struct Module {
    pub magic: String,
    pub version: u32,
    pub type_section: Option<Vec<FuncType>>,
    pub fn_section: Option<Vec<u32>>,
    pub code_section: Option<Vec<Function>>,
    pub export_section: Option<Vec<Export>>,
    pub import_section: Option<Vec<Import>>,
    pub memory_section: Option<Vec<Memory>>,

}

impl Default for Module {
    fn default() -> Self {
        Self { 
            magic: WASM_MAGIC.to_string(), version: 1, 
            type_section: None, fn_section: None, code_section: None, 
            export_section: None, import_section: None, 
            memory_section: None,
        }
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
                        SectionCode::Memory => {
                            let (_, memories) = decode_memory_section(section_contents)?;
                            module.memory_section = Some(memories);
                        }
                        SectionCode::Data => {
                            
                        }
                        SectionCode::Export => {
                            let (_, exports) = decode_export_section(section_contents)?;
                            module.export_section = Some(exports);
                        }
                        SectionCode::Import => {
                            let (_, imports) = decode_import_section(section_contents)?;
                            module.import_section = Some(imports);
                        }
                        SectionCode::Custom => {
                            // skip
                        }
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
        let(rest, tys) = decode_raw_seq(rest)?;
        let(_, params) = many0(decodea_value_type)(tys)?;
        
        // decode fn return types
        let(rest, tys) = decode_raw_seq(rest)?;
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
        Opcode::LocalSet => {
            let (input, i) = leb128_u32(input)?;
            Ok((input, Instruction::LocalSet(i)))
        }
        Opcode::I32Const => {
            let (input, value) = leb128_i32(input)?;
            Ok((input, Instruction::I32Const(value)))
        }
        Opcode::I32Store => {
            let (input, align) = leb128_u32(input)?;
            let (input, offset) = leb128_u32(input)?;
            Ok((input, Instruction::I32Store { align, offset }))
        }
        Opcode::I32Add => Ok((input, Instruction::I32Add)),
        Opcode::Call => {
            let (input, i) = leb128_u32(input)?;
            Ok((input, Instruction::Call(i)))
        }
    }
}

fn decode_raw_seq(input: &[u8]) -> IResult<&[u8], &[u8]> {
    let (rest, sz) = leb128_u32(input)?;
    let (rest, bytes) = take(sz)(rest)?;

    Ok((rest, bytes))
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
        let (rest, contents_body) = decode_raw_seq(input)?;
        let (_, bodies) = decode_function_body(contents_body)?;

        fns.push(bodies);

        input = rest;
    }

    Ok((input, fns))
}

fn decode_name(input: &[u8]) -> IResult<&[u8], String> {
    let (rest, name_bytes) = decode_raw_seq(input)?;
    
    Ok((rest, String::from_utf8(name_bytes.to_vec()).expect("Invalid utf8 sequence")))
}

fn decode_export_section(input: &[u8]) -> IResult<&[u8], Vec<Export>> {
    let (input, count) = leb128_u32(input)?;
    let mut exports = vec![];

    let mut remaining = input;

    for _ in 0..count {
        let (rest, name) = decode_name(remaining)?;
        let (rest, kind) = le_u8(rest)?;
        let (rest, i) = leb128_u32(rest)?;

        match kind {
            0 => {
                exports.push(Export { 
                    name, 
                    desc: ExportDesc::Func(i) 
                });
            }
            _ => unimplemented!("Unsupported export kind: {kind:X}")
        };

        remaining = rest;
    }

    Ok((remaining, exports))
}

fn decode_import_section(input: &[u8]) -> IResult<&[u8], Vec<Import>> {
    let (input, count) = leb128_u32(input)?;
    let mut imports = vec![];

    let mut remaining = input;

    for _ in 0..count {
        let (rest, mod_name) = decode_name(remaining)?;
        let (rest, field_name) = decode_name(rest)?;
        let (rest, kind) = le_u8(rest)?;
        let (rest, i) = leb128_u32(rest)?;

        match kind {
            0 => {
                imports.push(Import { mod_name, field_name, desc: ImportDesc::Func(i) })
            }
            _ => unimplemented!("Unsupported import kind: {kind:X}")
        }

        remaining = rest;
    }

    Ok((remaining, imports))
}

fn decode_memory_section(input: &[u8]) -> IResult<&[u8], Vec<Memory>> {
    let (input, count) = leb128_u32(input)?;
    let mut memories = vec![];

    let mut remaining = input;

    for _ in 0..count {
        let (rest, has_max) = le_u8(remaining)?;
        let (rest, initial) = leb128_u32(rest)?;

        let (rest, mem) = match has_max {
            0 => {
                (rest, Memory { initial: initial, limit: None })
            }
            1 => {
                let (rest, max) = leb128_u32(rest)?;
                (rest, Memory { initial: initial, limit: Some(max) })
            }
            _ => unreachable!(),
        };

        memories.push(mem);

        remaining = rest;
    }

    Ok((remaining, memories))
}

fn decode_expr(input: &[u8]) -> IResult<&[u8], u32> {
    let (rest, _) = leb128_u32(input)?;
    let (rest, offset) = leb128_u32(rest)?;
    let (rest, _) = leb128_u32(rest)?;

    Ok((rest, offset))
}

fn decode_data_section(input: &[u8]) -> IResult<&[u8], Vec<Data>> {
    let (input, count) = leb128_u32(input)?;
    let mut entries = vec![];

    let mut remaining = input;

    for _ in 0..count {
        let (rest, page) = leb128_u32(remaining)?;
        let (rest, offset) = decode_expr(rest)?;
        let (rest, bytes) = decode_raw_seq(rest)?;

        entries.push(Data { page, offset, bytes: Vec::from(bytes) });

        remaining = rest;
    }

    Ok((remaining, entries))
}

#[cfg(test)]
mod decoder_tests {
    use crate::binary::{instruction::Instruction, module::Module, section::{Function, FunctionLocal, SectionCode}, types::{Data, Export, ExportDesc, FuncType, Import, ImportDesc, Memory, ValueType}};
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
    fn decode_simplest_fn_exported() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (func $dummy) (export "dummy" (func $dummy)))"#)?;
        let module = Module::new(&wasm)?;

        let expected = Module {
            type_section: Some(vec![FuncType::default()]),
            fn_section: Some(vec![0]),
            code_section: Some(vec![Function { locals: vec![], code: vec![Instruction::End] }]),
            export_section: Some(vec![
                Export { name: "dummy".to_string(), desc: ExportDesc::Func(0) }
            ]),
            ..Default::default()
        };

        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_simplest_fn_imported() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (func $dummy (import "env" "dummy") (param i32) (result i32)))"#)?;
        let module = Module::new(&wasm)?;

        let expected = Module {
            type_section: Some(vec![FuncType { params: vec![ValueType::I32], returns: vec![ValueType::I32] }]),
            fn_section: None,
            code_section: None,
            import_section: Some(vec![
                Import { mod_name: "env".to_string(), field_name: "dummy".to_string(), desc: ImportDesc::Func(0) }
            ]),
            ..Default::default()
        };

        assert_eq!(expected, module);
        Ok(())
    }

    #[test]
    fn decode_alloc_memory() -> Result<()> {
        let wasm = wat::parse_str(r#"(module (memory 2 3)(memory 1))"#)?;
        let module = Module::new(&wasm)?;

        let expected = Module {
            memory_section: Some(vec![
                Memory { initial: 2, limit: Some(3) },
                Memory { initial: 1, limit: None },
            ]),
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
    fn decode_value_types() -> Result<()> {
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
        assert_eq!(Instruction::Call(2), super::decode_instruction(&[0x10, 0x02])?.1);
        assert_eq!(Instruction::LocalSet(2), super::decode_instruction(&[0x21, 2])?.1);
        assert_eq!(Instruction::I32Const(42), super::decode_instruction(&[0x41, 42])?.1);
        assert_eq!(Instruction::I32Store { align: 0xa, offset: 0x7f }, super::decode_instruction(&[0x36, 0xa, 0x7f])?.1);
        
        Ok(())
    }

    #[test]
    fn decode_export_sections() -> Result<()> {
        let expected = vec![
            Export { name: "dummy".to_string(), desc: ExportDesc::Func(0) }
        ];

        assert_eq!(expected, super::decode_export_section(&[0x01, 0x05, 0x64, 0x75, 0x6d, 0x6d, 0x79, 0, 0])?.1);
        Ok(())
    }

    #[test]
    fn decode_import_sections() -> Result<()> {
        let expected = vec![
            Import { mod_name: "env".to_string(), field_name: "add".to_string(), desc: ImportDesc::Func(1) }
        ];

        assert_eq!(expected, super::decode_import_section(&[0x01, 0x03, 0x65, 0x6e, 0x76, 0x03, 0x61, 0x64, 0x64, 0, 1])?.1);
        Ok(())
    }

    #[test]
    fn decode_memory_sections() -> Result<()> {
        let expected = vec![
            Memory { initial: 2, limit: Some(3) },
            Memory { initial: 1, limit: None },
        ];
        
        assert_eq!(expected, super::decode_memory_section(&[0x02, 0x01, 0x02, 0x03, 0, 0x01, 0x02])?.1);
        Ok(())
    }

    #[test]
    fn decode_data_sections() -> Result<()> {
        let expected = vec![
            Data { page: 0, offset: 0, bytes: Vec::from("Hello\n") }
        ];

        assert_eq!(expected, super::decode_data_section(&[0x01, 0, 0x41, 0, 0x0b, 0x06, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x0a, 0x0d])?.1);
        Ok(())
    }
}
