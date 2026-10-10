import json,subprocess,sys
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
cases=[]
for tokens in (4,16,256,1024,8192):
 for steps in (2,4,20,50):
  model=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True,shift_terminal=.02,base_image_seq_len=256,max_image_seq_len=8192,base_shift=.5,max_shift=.9)
  mu=.5+(tokens-256)*(.9-.5)/(8192-256);model.set_timesteps(steps,sigmas=np.linspace(1.,1/steps,steps),mu=mu)
  actual=np.array(json.loads(subprocess.check_output([sys.argv[1],str(tokens),str(steps)])));expected=model.sigmas.numpy();np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7);cases.append(dict(tokens=tokens,steps=steps,max_absolute_error=float(np.max(np.abs(actual-expected)))))
for args in [('4','1'),('0','4'),('4','101')]:assert subprocess.run([sys.argv[1],*args],capture_output=True).returncode!=0
print(json.dumps(dict(case_count=len(cases),cases=cases),indent=2))
