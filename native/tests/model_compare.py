"""Hand-authored capture validation cases; no fake/native numerical oracle."""
import struct
import subprocess


def verify(binary, root):
    expected,actual=root/'compare-expected.bin',root/'compare-actual.bin'
    def capture(values=(4.,2.,1.), selected=0, eos=0, progress=1):
        return struct.pack('<8I3fI',0x4b4d4331,1,3,1,2,selected,eos,progress,*values,0x444f4e45)
    baseline=capture()
    def run(left,right,accepted=False,absolute='0',relative='0'):
        expected.write_bytes(left);actual.write_bytes(right)
        result=subprocess.run([binary,str(expected),str(actual),absolute,relative],capture_output=True,timeout=5)
        assert result.returncode==(0 if accepted else 1), (result.stdout,result.stderr)
    run(baseline,baseline,True)
    for value in (float('nan'),float('inf'),-float('inf')):
        bad=capture((4.,value,1.));run(baseline,bad);run(bad,baseline)
    # Wrong greedy ID, including incorrect higher-ID selection on an exact tie.
    for bad in (capture(selected=1),capture((4.,4.,1.),selected=1),
                capture(eos=1),capture(eos=2),capture(progress=0),capture(progress=2),
                baseline+b'X',baseline[:-1],baseline[:-4]+b'FAIL'):
        run(baseline,bad);run(bad,baseline)
    run(capture((4.,4.,1.)),capture((4.,4.,1.)),True)
    # Exact representable boundaries and the immediately next float outside.
    def next_float(x):
        u=struct.unpack('<I',struct.pack('<f',x))[0]
        return struct.unpack('<f',struct.pack('<I',u+1))[0]
    for absolute,relative,boundary in (('0.25','0',2.25),('0','0.125',2.25),('0.125','0.125',2.375)):
        run(baseline,capture((4.,boundary,1.)),True,absolute,relative)
        run(baseline,capture((4.,next_float(boundary),1.)),False,absolute,relative)
    for invalid in ('nan','inf','-1','0x1','1junk'):
        run(baseline,baseline,False,invalid,'0');run(baseline,baseline,False,'0',invalid)
    # Valid bounded records with a different input ID / shape still mismatch.
    wrong=bytearray(baseline);struct.pack_into('<I',wrong,16,1);run(baseline,wrong)
    for offset,value in ((0,0),(4,2),(8,0),(8,262145),(12,0),(12,9),(20,3)):
        wrong=bytearray(baseline);struct.pack_into('<I',wrong,offset,value);run(baseline,wrong)
    print('Capture comparator: nonfinite, malformed records, greedy/ties, EOS/progress, trailing bytes and tolerance boundaries passed.')
