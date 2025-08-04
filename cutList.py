"""
classes to store all the analysis region definition
31/1/19
"""

import sys


class cutList(object):
    def __init__(self, Selection):
        self.cutsDict = {}
        self.IsoSigWeight = "1"
        self.PromptOriginWeight = "1"
        self.FakeOriginWeight = "1"
        self.badevents = "(1.)"

        self.IsoSig = "(((isSignalLep1==1 && isSignalLep2==1 && nlep==2) || (nlep==3 && isSignalLep1==1 && isSignalLep2==1 && isSignalLep3==1) || (isSignalLep1==1 && isSignalLep2==1 && isSignalLep3==1 && isSignalLep4==1 && nlep==4)) && LeptonTrigger)"
        self.MediumID = "((isMediumlep1==1 && isMediumlep2==1 && nlep==2) || (nlep==3 && isMediumlep1==1 && isMediumlep2==1 && isMediumlep3==1))"

        self.LooseVarLep1 = "((fabs(lepflav1)==11 && LepIsoLoose_VarRad1==1) || (fabs(lepflav1)==13 && LepIsoPflowLoose_VarRad1==1))"
        self.LooseVarLep2 = "((fabs(lepflav2)==11 && LepIsoLoose_VarRad2==1) || (fabs(lepflav2)==13 && LepIsoPflowLoose_VarRad2==1))"
        self.LooseVarLep3 = "((fabs(lepflav3)==11 && LepIsoLoose_VarRad3==1) || (fabs(lepflav3)==13 && LepIsoPflowLoose_VarRad3==1))"

        self.IsoLooseVar2 = (
            "(nlep==2 && " + self.LooseVarLep1 + " && " + self.LooseVarLep2 + ")"
        )
        self.IsoLooseVar3 = (
            "(nlep==3 && "
            + self.LooseVarLep1
            + " && "
            + self.LooseVarLep2
            + " && "
            + self.LooseVarLep3
            + ")"
        )

        self.IsoLooseVar = "(" + self.IsoLooseVar2 + " || " + self.IsoLooseVar3 + ")"

        self.MediumIDIsoLooseVar = "(" + self.MediumID + "&&" + self.IsoLooseVar + ")"

        self.Prompt1 = "((((IFFTypeLep1==0 || IFFTypeLep1==1 || IFFTypeLep1==2 || IFFTypeLep1==3 || IFFTypeLep1==6 || IFFTypeLep1==7))) || ((IFFTypeLep1==0 || IFFTypeLep1==1 || IFFTypeLep1==4|| IFFTypeLep1==7 || IFFTypeLep1==11)))"
        self.Prompt2 = "((((IFFTypeLep2==0 || IFFTypeLep2==1 || IFFTypeLep2==2 || IFFTypeLep2==3 || IFFTypeLep2==6 || IFFTypeLep2==7))) || ((IFFTypeLep2==0 || IFFTypeLep2==1 || IFFTypeLep2==4|| IFFTypeLep2==7 || IFFTypeLep2==11)))"
        self.Prompt3 = "((((IFFTypeLep3==0 || IFFTypeLep3==1 || IFFTypeLep3==2 || IFFTypeLep3==3 || IFFTypeLep3==6 || IFFTypeLep3==7))) || ((IFFTypeLep3==0 || IFFTypeLep3==1 || IFFTypeLep3==4|| IFFTypeLep3==7 || IFFTypeLep3==11)))"
        self.Prompt4 = "((((IFFTypeLep4==0 || IFFTypeLep4==1 || IFFTypeLep4==2 || IFFTypeLep4==3 || IFFTypeLep4==6 || IFFTypeLep4==7))) || ((IFFTypeLep4==0 || IFFTypeLep4==1 || IFFTypeLep4==4|| IFFTypeLep4==7 || IFFTypeLep4==11)))"

        self.IsoPromptLep2 = "(nlep==2 && " + self.Prompt1 + " && " + self.Prompt2 + ")"
        self.IsoPromptLep3 = (
            "(nlep==3 && "
            + self.Prompt1
            + " && "
            + self.Prompt2
            + " && "
            + self.Prompt3
            + ")"
        )
        self.IsoPromptLep4 = (
            "(nlep==4 && "
            + self.Prompt1
            + " && "
            + self.Prompt2
            + " && "
            + self.Prompt3
            + " && "
            + self.Prompt4
            + ")"
        )

        self.isPrompt = (
            "("
            + self.IsoPromptLep2
            + " || "
            + self.IsoPromptLep3
            + " || "
            + self.IsoPromptLep4
            + ")"
        )
        # self.isPrompt = "(true)"

        self.isnotPrompt1 = "(IFFTypeLep1!=2 || IFFTypeLep1!=3 || IFFTypeLep1!=6 || IFFTypeLep1!=7 || IFFTypeLep1!=4 || IFFTypeLep1!=11)"
        self.isnotPrompt2 = "(IFFTypeLep2!=2 || IFFTypeLep2!=3 || IFFTypeLep2!=6 || IFFTypeLep2!=7 || IFFTypeLep2!=4 || IFFTypeLep2!=11)"
        self.isnotPrompt3 = "(IFFTypeLep3!=2 || IFFTypeLep3!=3 || IFFTypeLep3!=6 || IFFTypeLep3!=7 || IFFTypeLep3!=4 || IFFTypeLep3!=11)"
        self.isnotPrompt4 = "(IFFTypeLep4!=2 || IFFTypeLep4!=3 || IFFTypeLep4!=6 || IFFTypeLep4!=7 || IFFTypeLep4!=4 || IFFTypeLep4!=11)"

        self.IsoNOPromptLep2 = (
            "(nlep==2 && (" + self.isnotPrompt1 + " || " + self.isnotPrompt2 + "))"
        )
        self.IsoNOPromptLep3 = (
            "(nlep==3 && ("
            + self.isnotPrompt1
            + " || "
            + self.isnotPrompt2
            + " || "
            + self.isnotPrompt3
            + "))"
        )
        self.IsoNOPromptLep4 = (
            "(nlep==4 && ("
            + self.isnotPrompt1
            + " || "
            + self.isnotPrompt2
            + " || "
            + self.isnotPrompt3
            + " || "
            + self.isnotPrompt4
            + "))"
        )

        self.isNOPrompt = (
            "("
            + self.IsoNOPromptLep2
            + " || "
            + self.IsoNOPromptLep3
            + " || "
            + self.IsoNOPromptLep4
            + ")"
        )
        self.FakeOriginWeight = (
            "(((nlep==2 && (AllOrigin==0 || (AllOrigin==1 &&  "
            + self.isnotPrompt1
            + " && "
            + self.isnotPrompt2
            + "))) || (nlep==3 && (AllOrigin3L==0 || (AllOrigin3L==1 && "
            + self.isnotPrompt1
            + " && "
            + self.isnotPrompt2
            + " && "
            + self.isnotPrompt3
            + " ))) || (nlep==4 && (AllOrigin4L==0 || (AllOrigin4L==1 && "
            + self.isnotPrompt1
            + " && "
            + self.isnotPrompt2
            + "&& "
            + self.isnotPrompt3
            + "&& "
            + self.isnotPrompt4
            + ")))))"
        )

        if Selection == "Stop3b":
            self.IsoSigWeight = self.IsoSig
            self.PromptOriginWeight = self.isPrompt
            self.BadEvents = self.badevents

            ###regions
            # commonSRselectionDF_3b = "(nlep==2 && isOS && !isSF) && (mll > 20) && (MDR > 75) && (lep1pT>25 && lep2pT>20)"
            # commonSRselectionSF_3b = "(nlep==2 && isOS && isSF && (mll<71.2 || mll>111.2)) && (mll > 20 && MDR > 75) && (lep1pT>25 && lep2pT>20)"
            commonSRselectionDF_3b = (
                "(nlep==2 && isOS && !isSF) " + "&&" + "(" + self.IsoSig + ")"
            )
            commonSRselectionSF_3b = (
                "(nlep==2 && isOS && isSF && (mll<71.2 || mll>111.2)) "
                + "&&"
                + "("
                + self.IsoSig
                + ")"
            )
            self.cutsDict["PreselectionDF"] = (
                commonSRselectionDF_3b + "&&" + "(" + self.IsoSig + ")"
            )
            self.cutsDict["PreselectionSF"] = (
                commonSRselectionSF_3b + "&&" + "(" + self.IsoSig + ")"
            )
            self.cutsDict["Preselection"] = (
                "(("
                + commonSRselectionDF_3b
                + ") || ("
                + commonSRselectionSF_3b
                + "))"
                + "&&"
                + "("
                + self.IsoSig
                + ")"
            )
            self.cutsDict["Preselection3Bodyee"] = (
                "(("
                + commonSRselectionDF_3b
                + ") || ("
                + commonSRselectionSF_3b
                + "))  && (lepflav1*lepflav2==-121)"
            )
            self.cutsDict["Preselection3Bodymumu"] = (
                "(("
                + commonSRselectionDF_3b
                + ") || ("
                + commonSRselectionSF_3b
                + ")) &&(lep1pT>25 && lep2pT>20) && (lepflav1*lepflav2==-169)"
            )
            self.cutsDict["Preselection3Bodyemu"] = (
                "(("
                + commonSRselectionDF_3b
                + ") || ("
                + commonSRselectionSF_3b
                + ")) && (lepflav1*lepflav2==-143)"
            )
            self.cutsDict["Preselection3BodySF+b-veto"] = (
                commonSRselectionSF_3b + "&& (nbjet==0)"
            )
            self.cutsDict["Preselection3BodyDF+b-veto"] = (
                commonSRselectionDF_3b + "&& (nbjet==0)"
            )
            self.cutsDict["Preselection3BodySF+b-jets"] = (
                commonSRselectionSF_3b + "&& (nbjet>0)"
            )
            self.cutsDict["Preselection3BodyDF+b-jets"] = (
                commonSRselectionDF_3b + "&& (nbjet>0)"
            )
            self.cutsDict["SRWDF"] = (
                "("
                + commonSRselectionDF_3b
                + "&& (DPB_vSS > 2.3 && RPT > 0.78 && METsig > 12 && MDR > 105 && gamInvRp1 > 0.7 && nbjet==0)"
                + ")"
            )
            self.cutsDict["SRWSF"] = (
                "("
                + commonSRselectionSF_3b
                + "&& (DPB_vSS > 2.3 && RPT > 0.78 && METsig > 12 && MDR > 105 && gamInvRp1 > 0.7 && nbjet==0)"
                + ")"
            )
            self.cutsDict["SRTDF"] = (
                "("
                + commonSRselectionDF_3b
                + "&& (DPB_vSS > 2.3 && RPT > 0.7  && METsig > 12 && MDR > 120 && gamInvRp1 > 0.7 && nbjet > 0)"
                + ")"
            )
            self.cutsDict["SRTSF"] = (
                "("
                + commonSRselectionSF_3b
                + "&& (DPB_vSS > 2.3 && RPT > 0.7  && METsig > 12 && MDR > 120 && gamInvRp1 > 0.7 && nbjet > 0)"
                + ")"
            )

            # ML regions
            # presel_ml = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 100 && isOS) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 65.2 || mll > 115.2) || !isSF))"
            presel_ml = (
                "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS && METsig > 10) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF) && (RPT > 0.70))"
                + "&&"
                + "("
                + self.IsoSig
                + ")"
            )
            self.cutsDict["PreselML"] = presel_ml + "&&" + "(" + self.IsoSig + ")"
            # Prelim SRs
            # self.cutsDict["SRsmallDM0b"] = presel_ml  + " && (nbjet == 0) && (DNN_smallDM_bjets0_sig >= 0.96)"
            # self.cutsDict["SRsmallDM1b"] = presel_ml + " && (nbjet > 0) && (DNN_smallDM_bjets1_sig >= 0.96)"
            # self.cutsDict["SRlargeDM0b"] = presel_ml + " && (nbjet == 0) && (DNN_largeDM_bjets0_sig > 0.96)"
            # self.cutsDict["SRlargeDM1b"] = presel_ml + " && (nbjet > 0) && (DNN_largeDM_bjets1_sig >= 0.96)"

            # Binned SRs -old
            # small0b_bins = [0.84, 0.91, 0.96, 0.98, 1]
            # small1b_bins = [0.97, 0.99, 1]
            # large0b_bins = [0.93, 0.96, 1]
            # large1b_bins = [0.93, 0.94, 0.97, 0.99, 1]

            # Binned SR -RPT METsig new
            small0b_bins = [0.60, 0.69, 0.82, 0.92, 0.972, 1]
            small1b_bins = [0.952, 0.977, 1]
            large0b_bins = [0.82, 0.855, 0.943, 1]
            large1b_bins = [0.85, 0.894, 0.917, 0.953, 1]

            # Run2
            # Commented values in R2 and R3 are before making left edge of two runs similar.
            # small0b_run2 = [0.760, 0.890, 0.957, 1]
            small0b_run2 = [0.748,  0.890, 0.957, 1]
            # small1b_run2 = [0.946, 0.966, 1]
            small1b_run2 = [0.939, 0.966, 1]
            # large0b_run2 = [0.942, 1]
            large0b_run2 = [0.856, 1]
            # large1b_run2 = [0.899, 0.958, 1]
            large1b_run2 = [0.897, 0.958, 1]

            # Run3
            small0b_run3 = [0.748, 0.798, 0.927, 1]
            small1b_run3 = [0.939, 1]
            large0b_run3 = [0.856, 0.926, 1]
            large1b_run3 = [0.897, 0.952, 1]

            # SRSF0b_bins = [0.651, 0.733, 0.835, 1]
            # SRSF1b_bins = [0.827, 0.937, 1]
            # SRDF0b_bins = [0.812, 0.903, 0.959, 1]
            # SRDF1b_bins = [0.757, 0.831, 0.957, 1]

            self.cutsDict["SRsmallDM0b0"] = (
                presel_ml
                + f" && (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_bins[0]} && DNN_smallDM_bjets0_sig<={small0b_bins[1]})"
            )
            self.cutsDict["SRsmallDM0b1"] = (
                presel_ml
                + f"&& (nbjet == 0) &&  (DNN_smallDM_bjets0_sig>{small0b_bins[1]} && DNN_smallDM_bjets0_sig<={small0b_bins[2]})"
            )
            self.cutsDict["SRsmallDM0b2"] = (
                presel_ml
                + f"&& (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_bins[2]} && DNN_smallDM_bjets0_sig<={small0b_bins[3]})"
            )
            self.cutsDict["SRsmallDM0b3"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_smallDM_bjets0_sig>{small0b_bins[3]} && DNN_smallDM_bjets0_sig<={small0b_bins[4]})"
            )
            self.cutsDict["SRsmallDM0b4"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_smallDM_bjets0_sig>{small0b_bins[4]} && DNN_smallDM_bjets0_sig<={small0b_bins[5]})"
            )
            self.cutsDict["SRsmallDM0bALL"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_smallDM_bjets0_sig>{small0b_bins[0]})"
            )

            self.cutsDict["SRsmallDM1b0"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_smallDM_bjets1_sig>{small1b_bins[0]} && DNN_smallDM_bjets1_sig<={small1b_bins[1]})"
            )
            self.cutsDict["SRsmallDM1b1"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_smallDM_bjets1_sig>{small1b_bins[1]} && DNN_smallDM_bjets1_sig<={small1b_bins[2]})"
            )
            self.cutsDict["SRsmallDM1bALL"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_smallDM_bjets1_sig>{small1b_bins[0]})"
            )

            self.cutsDict["SRlargeDM0b0"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_bins[0]} && DNN_largeDM_bjets0_sig<={large0b_bins[1]})"
            )
            self.cutsDict["SRlargeDM0b1"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_bins[1]} && DNN_largeDM_bjets0_sig<={large0b_bins[2]})"
            )
            self.cutsDict["SRlargeDM0b2"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_bins[2]} && DNN_largeDM_bjets0_sig<={large0b_bins[3]})"
            )
            self.cutsDict["SRlargeDM0bALL"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_bins[0]})"
            )

            self.cutsDict["SRlargeDM1b0"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_bins[0]} && DNN_largeDM_bjets1_sig<={large1b_bins[1]})"
            )
            self.cutsDict["SRlargeDM1b1"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_bins[1]} && DNN_largeDM_bjets1_sig<={large1b_bins[2]})"
            )
            self.cutsDict["SRlargeDM1b2"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_bins[2]} && DNN_largeDM_bjets1_sig<={large1b_bins[3]})"
            )
            self.cutsDict["SRlargeDM1b3"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_bins[3]} && DNN_largeDM_bjets1_sig<={large1b_bins[4]})"
            )
            self.cutsDict["SRlargeDM1bALL"] = (
                presel_ml + f"&&(DNN_largeDM_bjets1_sig>{large1b_bins[0]})"
            )

            # Run2 SRs
            self.cutsDict["SRRun2smallDM0b0"] = (
                presel_ml
                + f" && (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_run2[0]} && DNN_smallDM_bjets0_sig<={small0b_run2[1]})"
            )
            self.cutsDict["SRRun2smallDM0b1"] = (
                presel_ml
                + f" && (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_run2[1]} && DNN_smallDM_bjets0_sig<={small0b_run2[2]})"
            )
            self.cutsDict["SRRun2smallDM0b2"] = (
                presel_ml
                + f" && (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_run2[2]} && DNN_smallDM_bjets0_sig<={small0b_run2[3]})"
            )
            self.cutsDict["SRRun2smallDM0bALL"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_smallDM_bjets0_sig>{small0b_run2[0]})"
            )

            self.cutsDict["SRRun2smallDM1b0"] = (
                presel_ml
                + f" && (nbjet > 0) && (DNN_smallDM_bjets1_sig>{small1b_run2[0]} && DNN_smallDM_bjets1_sig<={small1b_run2[1]})"
            )
            self.cutsDict["SRRun2smallDM1b1"] = (
                presel_ml
                + f" && (nbjet > 0) && (DNN_smallDM_bjets1_sig>{small1b_run2[1]} && DNN_smallDM_bjets1_sig<={small1b_run2[2]})"
            )
            self.cutsDict["SRRun2smallDM1bALL"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_smallDM_bjets1_sig>{small1b_run2[0]})"
            )

            self.cutsDict["SRRun2largeDM0bALL"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_run2[0]})"
            )

            self.cutsDict["SRRun2largeDM1b0"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_run2[0]} && DNN_largeDM_bjets1_sig<={large1b_run2[1]})"
            )
            self.cutsDict["SRRun2largeDM1b1"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_run2[1]} && DNN_largeDM_bjets1_sig<={large1b_run2[2]})"
            )
            self.cutsDict["SRRun2largeDM1bALL"] = (
                presel_ml + f"&&(DNN_largeDM_bjets1_sig>{large1b_run2[0]})"
            )

            # Run3 SRs
            self.cutsDict["SRRun3smallDM0b0"] = (
                presel_ml
                + f" && (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_run3[0]} && DNN_smallDM_bjets0_sig<={small0b_run3[1]})"
            )
            self.cutsDict["SRRun3smallDM0b1"] = (
                presel_ml
                + f" && (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_run3[1]} && DNN_smallDM_bjets0_sig<={small0b_run3[2]})"
            )
            self.cutsDict["SRRun3smallDM0b2"] = (
                presel_ml
                + f" && (nbjet == 0) && (DNN_smallDM_bjets0_sig>{small0b_run3[2]} && DNN_smallDM_bjets0_sig<={small0b_run3[3]})"
            )
            self.cutsDict["SRRun3smallDM0bALL"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_smallDM_bjets0_sig>{small0b_run3[0]})"
            )

            self.cutsDict["SRRun3smallDM1bALL"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_smallDM_bjets1_sig>{small1b_run3[0]})"
            )

            self.cutsDict["SRRun3largeDM0b0"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_run3[0]} && DNN_largeDM_bjets0_sig<={large0b_run3[1]})"
            )
            self.cutsDict["SRRun3largeDM0b1"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_run3[1]} && DNN_largeDM_bjets0_sig<={large0b_run3[2]})"
            )
            self.cutsDict["SRRun3largeDM0bALL"] = (
                presel_ml
                + f"&& (nbjet == 0) &&(DNN_largeDM_bjets0_sig>{large0b_run3[0]})"
            )

            self.cutsDict["SRRun3largeDM1b0"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_run3[0]} && DNN_largeDM_bjets1_sig<={large1b_run3[1]})"
            )
            self.cutsDict["SRRun3largeDM1b1"] = (
                presel_ml
                + f"&& (nbjet > 0) &&(DNN_largeDM_bjets1_sig>{large1b_run3[1]} && DNN_largeDM_bjets1_sig<={large1b_run3[2]})"
            )
            self.cutsDict["SRRun3largeDM1bALL"] = (
                presel_ml + f"&&(DNN_largeDM_bjets1_sig>{large1b_run3[0]})"
            )
            # CRs
            self.cutsDict["CRTOPsmall1bRun2"] = (
                presel_ml + "&& (nbjet > 0) && (DNN_smallDM_bjets1_sig <= 0.942) && (DNN_smallDM_bjets1_ttbar > 0.50)"
            )
            self.cutsDict["CRTOPlarge1bRun2"] = (
                presel_ml + "&& (nbjet > 0) && (DNN_largeDM_bjets1_sig <= 0.67) && (DNN_largeDM_bjets1_ttbar > 0.50)"
            )
            self.cutsDict["CRVVsmall0bRun2"] = (
                presel_ml + "&& (nbjet == 0) && (DNN_smallDM_bjets0_sig <= 0.59) && (DNN_smallDM_bjets0_VV > 0.60)"
            )
            self.cutsDict["CRVVlarge0bRun2"] = (
                presel_ml + "&& (nbjet == 0) && (DNN_largeDM_bjets0_sig <= 0.59) && (DNN_largeDM_bjets0_VV > 0.60)"
            )

            self.cutsDict["CRTOPsmall1bRun3"] = (
                presel_ml + "&& (nbjet > 0) && (DNN_smallDM_bjets1_sig <= 0.899) && (DNN_smallDM_bjets1_ttbar > 0.50)"
            )
            self.cutsDict["CRTOPlarge1bRun3"] = (
                presel_ml + "&& (nbjet > 0) && (DNN_largeDM_bjets1_sig <= 0.487) && (DNN_largeDM_bjets1_ttbar > 0.50)"
            )
            self.cutsDict["CRVVsmall0bRun3"] = (
                presel_ml + "&& (nbjet == 0) && (DNN_smallDM_bjets0_sig <= 0.248) && (DNN_smallDM_bjets0_VV > 0.60)"
            )
            self.cutsDict["CRVVlarge0bRun3"] = (
                presel_ml + "&& (nbjet == 0) && (DNN_smallDM_bjets0_sig <= 0.446) && (DNN_largeDM_bjets0_VV > 0.60)"
            )

            # SF/DF Model
            """
            SRSF0b_bins = [0.651, 0.733, 0.835, 1]
            SRSF1b_bins = [0.827, 0.937, 1]
            SRDF0b_bins = [0.812, 0.903, 0.959, 1]
            SRDF1b_bins = [0.757, 0.831, 0.957, 1]
            presel_sfdf = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2) || !isSF))"
            self.cutsDict["Preselection"] = presel_sfdf
            presel_sf =  "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS && METsig>10) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (isSF && (mll < 71.2 || mll > 111.2)))"
            presel_df = "((nlep == 2 && lep1pT > 25 && lep2pT > 20 && mll > 20 &&  MDR > 75 && isOS && METsig>10) && (isSignalLep1 == 1 && isSignalLep2 == 1) && (!isSF))"
            self.cutsDict["PreselectionSF"] = presel_sf
            self.cutsDict["PreselectionDF"] = presel_df
            self.cutsDict["SRSF0b"] = "(nbjet == 0) && (DNN_SF0b_sig>0.85) && " + presel_sf
            self.cutsDict["SRSF1b"] = "(nbjet > 0) && (DNN_SF1b_sig>0.95) && " + presel_sf
            self.cutsDict["SRDF0b"] = "(nbjet == 0) && (DNN_DF0b_sig>0.9) && " + presel_df
            self.cutsDict["SRDF1b"] = "(nbjet > 0) && (DNN_DF1b_sig>0.95) && " + presel_df
            
            self.cutsDict["SRSF0b0"] = presel_sf + f" && (nbjet == 0) && (DNN_SF0b_sig>{SRSF0b_bins[0]} && DNN_SF0b_sig<={SRSF0b_bins[1]})"
            self.cutsDict["SRSF0b1"] = presel_sf + f" && (nbjet == 0) && (DNN_SF0b_sig>{SRSF0b_bins[1]} && DNN_SF0b_sig<={SRSF0b_bins[2]})"
            self.cutsDict["SRSF0b2"] = presel_sf + f" && (nbjet == 0) && (DNN_SF0b_sig>{SRSF0b_bins[2]} && DNN_SF0b_sig<={SRSF0b_bins[3]})"
            
            self.cutsDict["SRSF1b0"] = presel_sf + f" && (nbjet > 0) && (DNN_SF1b_sig>{SRSF1b_bins[0]} && DNN_SF1b_sig<={SRSF1b_bins[1]})"
            self.cutsDict["SRSF1b1"] = presel_sf + f" && (nbjet > 0) && (DNN_SF1b_sig>{SRSF1b_bins[1]} && DNN_SF1b_sig<={SRSF1b_bins[2]})"
            
            self.cutsDict["SRDF0b0"] = presel_df + f" && (nbjet == 0) && (DNN_DF0b_sig>{SRDF0b_bins[0]} && DNN_DF0b_sig<={SRDF0b_bins[1]})"
            self.cutsDict["SRDF0b1"] = presel_df + f" && (nbjet == 0) && (DNN_DF0b_sig>{SRDF0b_bins[1]} && DNN_DF0b_sig<={SRDF0b_bins[2]})"
            self.cutsDict["SRDF0b2"] = presel_df + f" && (nbjet == 0) && (DNN_DF0b_sig>{SRDF0b_bins[2]} && DNN_DF0b_sig<={SRDF0b_bins[3]})"
            
            self.cutsDict["SRDF1b0"] = presel_df + f" && (nbjet > 0) && (DNN_DF1b_sig>{SRDF1b_bins[0]} && DNN_DF1b_sig<={SRDF1b_bins[1]})"
            self.cutsDict["SRDF1b1"] = presel_df + f" && (nbjet > 0) && (DNN_DF1b_sig>{SRDF1b_bins[1]} && DNN_DF1b_sig<={SRDF1b_bins[2]})"
            self.cutsDict["SRDF1b2"] = presel_df + f" && (nbjet > 0) && (DNN_DF1b_sig>{SRDF1b_bins[2]} && DNN_DF1b_sig<={SRDF1b_bins[3]})"
            
            #PrelimCRs
            self.cutsDict["CRTOPSF0b"] = presel_sf + f" && (nbjet == 0) && (DNN_SF0b_sig <0.651) && (DNN_SF0b_ttbar > 0.2)"
            self.cutsDict["CRTOPSF1b"] = presel_sf + f" && (nbjet > 0) && (DNN_SF1b_sig <0.827) && (DNN_SF1b_ttbar > 0.2)"
            self.cutsDict["CRTOPDF0b"] = presel_sf + f" && (nbjet == 0) && (DNN_DF0b_sig <0.812) && (DNN_DF0b_ttbar > 0.2)"
            self.cutsDict["CRTOPDF1b"] = presel_sf + f" && (nbjet > 0) && (DNN_DF1b_sig <0.757) && (DNN_DF1b_ttbar > 0.2)"
            
            self.cutsDict["CRVVSF0b"] = presel_sf + f" && (nbjet == 0) && (DNN_SF0b_sig <0.651) && (DNN_SF0b_VV > 0.2)"
            self.cutsDict["CRVVDF0b"] = presel_sf + f" && (nbjet == 0) && (DNN_DF0b_sig <0.812) && (DNN_SF0b_VV > 0.2)"
            """

        else:
            print("Wrong analysis selection: only Stop3b and can be chosen")
            sys.exit(-1)
