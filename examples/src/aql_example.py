from qlauncher.launcher.aql import AQLManager
from qlauncher.problems import EC, MaxCut
from qlauncher.routines.dwave import DwaveSolver, SimulatedAnnealingBackend

with AQLManager() as launcher:
    launcher.add(backend=SimulatedAnnealingBackend(), algorithm=DwaveSolver(1), problem=EC('quadratic', instance_name='micro'))
    launcher.add_algorithm(DwaveSolver(2), times=2)
    launcher.add_problem(MaxCut(instance_name='default'), times=3)
    result = launcher.result
    result_bitstring = launcher.result_bitstring
    # When quiting from task addition part all added problems starts to run
    # result list will get new values if possible
    # to get values you need to exit from context manager and then wait until calculation process stops
    # If you want to execute your own code while the results are in progress you should use multithreaded option (which will be added soon) or you can add it as 'algorithm' which will be passed

    # overall it is tool if you want to quickly run multiple problems on real device, and don't want to wait after each one ends, or start 30 python programs
print(result)
