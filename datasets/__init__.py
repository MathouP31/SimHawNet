def create_dataset(name, data_dir="data_dir",max_event =None, Graph=None, T_max =None, list_edges =None,**hparams):
    from .features import create_feature

    if name == "highschool":
        from .highschool import Highschool
        return Highschool(T_max = T_max)
    
    if name == "reality":
        from .reality import Reality
        return Reality(T_max = T_max)
    
    if name == "mid":
        from .mid import MID
        return MID(T_max = T_max)

    if name == "college_msg":
        from .college_msg import CollegeMsg
        return CollegeMsg(data_path=data_dir, max_event = max_event, **hparams)

    if name == "enron":
        from .enron import Enron
        return Enron(data_path=data_dir, max_event= max_event,T_max = T_max, **hparams)
    
    if name == "toy":
        from .toy import ToyDataset
        return ToyDataset(**hparams)

    if name == "simul":
        from .simul import SimulDataset
        return SimulDataset(data_dir, **hparams)
    
    if name =="simulation":
        from .simulation import SimulationDataset
        return SimulationDataset(Graph, T_max,**hparams)
    
    if name =="simulation_simple":
        from .simulation_simple import SimulationDatasetSimple
        return SimulationDatasetSimple(Graph,T_max, **hparams)
    
    if name == "affichage":
        from .affichage  import Affichage
        return Affichage(list_edges, **hparams)

    




    
