# /afs/cern.ch/user/s/skandel/skandel/Run3ttMET/DNN/datanewmodel
# python generate_inference_bjet.py -m large -n 1 -e odd
def generate_inference(model_file):
    import ROOT
    ROOT.TMVA.PyMethodBase.PyInitialize()
    model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(model_file)
    model.Generate()
    model.OutputGenerated(f"model_{mass_diff}DM_bjet{nbjet}_{event_type}.hxx")
    return

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mass_diff", type=str, default='small', choices=["small", "large"],  help="Mass difference between signal and background")
    #parser.add_argument("-c", "--campaign", type=str, default="run2", help="Campaign to use")
    parser.add_argument("-e", "--event_type", type=str, default="odd", choices=["odd", "even"], help="Event type to use")
    parser.add_argument("-n", "--nbjet", type=int, default=0, choices=[0,1], help="Number of bjets")
    
    args = parser.parse_args()
    mass_diff = args.mass_diff
    event_type = args.event_type
    nbjet = args.nbjet
    model_file = f"model_out/model_{mass_diff}DM_bjet{nbjet}_{event_type}.h5"
    
    generate_inference(model_file)
    os.system(f"mv model_{mass_diff}DM_bjet{nbjet}_{event_type}.hxx model_files/inference_files/")
    os.system(f"mv model_{mass_diff}DM_bjet{nbjet}_{event_type}.dat model_files/inference_files/")
    print("Done!")
