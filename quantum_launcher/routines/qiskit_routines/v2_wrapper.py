import time
from qiskit.primitives.base import BaseSamplerV1, BaseSamplerV2
from qiskit.primitives import  SamplerResult, BasePrimitiveJob
from qiskit import transpile
    
class DummyJob(BasePrimitiveJob):
    '''
    Dummy data holder, to pass v1 result from V2Wrapper
    '''
    def __init__(self, job_id,res, **kwargs):
        super().__init__(job_id, **kwargs)
        self.res = res    
    
    def result(self):
        return self.res
    
    def cancel(self):
        pass
    
    def status(self):
        return "COMPLETED"
    
    def done(self):
        return True
    
    def cancelled(self):
        return False
    
    def running(self):
        return False
    
    def in_final_state(self):
        return True

    
class SamplerV2Adapter(BaseSamplerV1):
    """
    V1 adapter for V2 samplers.
    """
    def __init__(self, samplerv2:BaseSamplerV2,backend=None, options: dict | None = None):
        self.samplerv2 = samplerv2
        self.backend = backend
        super().__init__()
        
    def _run(self,circuits,parameter_values=None,**run_options):
        
        #Transpile qaoa circuit to backend instruction set, if backend is provided
        #TODO (or question): I pass a backend into SamplerV2 as *mode* but here samplerv2.mode returns None, why?
        if self.backend is not None:
            circuits = [transpile(circuit,self.backend) for circuit in circuits]
            
        v2_list = list(zip(circuits,parameter_values))
        job =  self.samplerv2.run(pubs=v2_list,**run_options)
        
        job_id = job.job_id()
        try:
            job.wait_for_final_state() #ibm runtime jobs have this nice method, e.g aer jobs don't
        except Exception as e:
            while not job.in_final_state():
                time.sleep(0.001)
        
        #v2 results have only counts available
        data = job.result()[0].data["meas"]
        counts = data.get_int_counts()
        quasi_dists = {k:v/data.num_shots for k,v in counts.items()}
        
        metadata = job.result()[0].metadata
        metadata["sampler_version"] = 2 #might be useful for debugging
        
        v1_result =  SamplerResult(quasi_dists=[quasi_dists],metadata=[metadata])
        
        #QAOA expects a job
        return DummyJob(job_id,v1_result)