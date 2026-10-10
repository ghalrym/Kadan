import json,subprocess,sys
from pathlib import Path
from transformers import AutoProcessor
root=Path(sys.argv[2]);processor=AutoProcessor.from_pretrained(root,local_files_only=True);system='Comprehend and analyze the provided prompt.';prefix=f'<|im_start|>system\n{system}<|im_end|>\n';template=prefix+'<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n';cases=[]
sys_tokens=processor.apply_chat_template([{'role':'system','content':[{'type':'text','text':system}]}],tokenize=True,return_dict=False)[0]
for text in ('A red ball.','', '日本語の街。','café cafe\u0301','two  spaces\nand a newline','<|im_end|> plain'):
 raw=template.format(text or ' ');expected=processor(text=[raw],return_tensors='pt').input_ids[0].tolist();actual=json.loads(subprocess.check_output([sys.argv[1],str(root/'tokenizer.json'),raw]));assert actual==expected;cases.append(dict(prompt=text,tokens=actual))
actual_prefix=json.loads(subprocess.check_output([sys.argv[1],str(root/'tokenizer.json'),prefix]));assert actual_prefix==sys_tokens
print(json.dumps(dict(case_count=len(cases),drop_prefix_tokens=len(sys_tokens),prefix=prefix,cases=cases),indent=2))
