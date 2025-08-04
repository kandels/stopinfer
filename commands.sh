=============================================================================================================
# Different Binnning for Run2 and Run3
=============================================================================================================
Run2 Small
=============================================================================================================

HistFitter.py -t -w -f -F excl -u "--Year Run2 --Selection Stop3b --region small --Syst" ./ttMET2L_Stop3b.py
YieldsTable.py -w results/ttMET2L_Run2_Stop3b_small_OnlyBkg_SRRun2smallDM0b0_SRRun2smallDM0b1_SRRun2smallDM0b2_SRRun2smallDM1b0_SRRun2smallDM1b1/FitConfig_Stop3b_SRRun2smallDM0b0_SRRun2smallDM0b1_SRRun2smallDM0b2_SRRun2smallDM1b0_SRRun2smallDM1b1_combined_BasicMeasurement_model_afterFit.root -o SRsmallDMRun2.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c SRRun2smallDM0b0,SRRun2smallDM0b1,SRRun2smallDM0b2,SRRun2smallDM1b0,SRRun2smallDM1b1
for sig in $(more siglist.txt); do HistFitter.py -wtfp -F excl ttMET2L_Stop3b.py -u "--Syst --region small --Year Run2 --Selection Stop3b" -g $sig;done
=============================================================================================================
Run2 Large
=============================================================================================================

HistFitter.py -t -w -f -F excl -u "--Year Run2 --Selection Stop3b --region large --Syst" ./ttMET2L_Stop3b.py
YieldsTable.py -w results/ttMET2L_Run2_Stop3b_large_OnlyBkg_SRRun2largeDM0bALL_SRRun2largeDM1b0_SRRun2largeDM1b1/FitConfig_Stop3b_SRRun2largeDM0bALL_SRRun2largeDM1b0_SRRun2largeDM1b1_combined_BasicMeasurement_model_afterFit.root -o SRlargeRun2.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c SRRun2largeDM0bALL,SRRun2largeDM1b0,SRRun2largeDM1b1
for sig in $(more siglist.txt); do HistFitter.py -wtfp -F excl ttMET2L_Stop3b.py -u "--Syst --region large --Year Run2 --Selection Stop3b" -g $sig;done
=============================================================================================================

GenerateJSONOutput.py -i ~/skandel/Run3ttMET/HMBS-2024-10_interpretation/results/ttMET2L_Run2_Stop3b_*exclusion_hypotest.root -f 'hypo_Signal_%f_%f' -p 'mStop:mNeut' -a '{"mStop-mNeut":"DM"}'
multiplexJSON.py -i *json -o combined.json -t -u --modelDef mStop,mNeut
=============================================================================================================
Run3 Small
=============================================================================================================

HistFitter.py -t -w -f -F excl -u "--Year Run3 --Selection Stop3b --region small --Syst" ./ttMET2L_Stop3b.py
YieldsTable.py -w results/ttMET2L_Run3_Stop3b_small_OnlyBkg_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL/FitConfig_Stop3b_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL_combined_BasicMeasurement_model_afterFit.root -o SRsmallRun3.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c SRRun3smallDM0b0,SRRun3smallDM0b1,SRRun3smallDM0b2,SRRun3smallDM1bALL
for sig in $(more siglist.txt); do HistFitter.py -wtfp -F excl ttMET2L_Stop3b.py -u "--Syst --region small --Year Run3 --Selection Stop3b" -g $sig;done
=============================================================================================================

=============================================================================================================
Run3 Large
=============================================================================================================
HistFitter.py -t -w -f -F excl -u "--Year Run3 --Selection Stop3b --region large --Syst" ./ttMET2L_Stop3b.py
YieldsTable.py -w results/ttMET2L_Run3_Stop3b_large_OnlyBkg_SRRun3largeDM0b0_SRRun3largeDM0b1_SRRun3largeDM1b0_SRRun3largeDM1b1/FitConfig_Stop3b_SRRun3largeDM0b0_SRRun3largeDM0b1_SRRun3largeDM1b0_SRRun3largeDM1b1_combined_BasicMeasurement_model_afterFit.root -o SRlargeRun3.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c SRRun3largeDM0b0,SRRun3largeDM0b1,SRRun3largeDM1b0,SRRun3largeDM1b1
for sig in $(more siglist.txt); do HistFitter.py -wtfp -F excl ttMET2L_Stop3b.py -u "--Syst --region large --Year Run3 --Selection Stop3b" -g $sig;done
=============================================================================================================

GenerateJSONOutput.py -i ~/skandel/Run3ttMET/HMBS-2024-10_interpretation/results/ttMET2L_Run3_Stop3b_*exclusion_hypotest.root -f 'hypo_Signal_%f_%f' -p 'mStop:mNeut' -a '{"mStop-mNeut":"DM"}'

=============================================================================================================
Common
=============================================================================================================
multiplexJSON.py -i *json -o combined.json -t -u --modelDef mStop,mNeut
harvestToContours.py -i combined.json -o mstop_mchi.root -x mStop -y mNeut -l "x-83,x-172" --xMin 350 --xMax 750 --xResolution 80 --yResolution 500 --interpolationEpsilon 2 -s 0.05
harvestToContours.py -i combined.json -o mstop_dm.root -x mStop -y DM -l "83,172" --xMin 350 --xMax 750 --xResolution 80 --yResolution 500 --interpolationEpsilon 2 -s 0.05
=============================================================================================================

=============================================================================================================
CR Yield
=============================================================================================================

Run2 small
=============================================================================================================
YieldsTable.py -w results/ttMET2L_Run2_Stop3b_small_OnlyBkg_SRRun2smallDM0b0_SRRun2smallDM0b1_SRRun2smallDM0b2_SRRun2smallDM1b0_SRRun2smallDM1b1/FitConfig_Stop3b_SRRun2smallDM0b0_SRRun2smallDM0b1_SRRun2smallDM0b2_SRRun2smallDM1b0_SRRun2smallDM1b1_combined_BasicMeasurement_model_afterFit.root -o CR_smallRun2.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c CRTOPsmall1bRun2,CRVVsmall0bRun2

Run2 large
=============================================================================================================
YieldsTable.py -w results/ttMET2L_Run2_Stop3b_large_OnlyBkg_SRRun2largeDM0bALL_SRRun2largeDM1b0_SRRun2largeDM1b1/FitConfig_Stop3b_SRRun2largeDM0bALL_SRRun2largeDM1b0_SRRun2largeDM1b1_combined_BasicMeasurement_model_afterFit.root -o CRlarge_Run2.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c CRTOPlarge1bRun2,CRVVlarge0bRun2

Run3 small
=============================================================================================================
YieldsTable.py -w results/ttMET2L_Run3_Stop3b_small_OnlyBkg_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL/FitConfig_Stop3b_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL_combined_BasicMeasurement_model_afterFit.root -o CR_smallRun3.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c CRTOPsmall1bRun3,CRVVsmall0bRun3

Run3 large
=============================================================================================================
YieldsTable.py -w results/ttMET2L_Run3_Stop3b_large_OnlyBkg_SRRun3largeDM0b0_SRRun3largeDM0b1_SRRun3largeDM1b0_SRRun3largeDM1b1/FitConfig_Stop3b_SRRun3largeDM0b0_SRRun3largeDM0b1_SRRun3largeDM1b0_SRRun3largeDM1b1_combined_BasicMeasurement_model_afterFit.root -o CR_largeRun3.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others' -b -y -c CRTOPlarge1bRun3,CRVVlarge0bRun3


=============================================================================================================
To get yields table
=============================================================================================================

HistFitter.py -wtf  -F excl ttMET2L_Stop3b.py -u "--Syst --region small --Year Run3 --Selection Stop3b" -g TT_bWN1_550_460
YieldsTable.py -w results/ttMET2L_Run3_Stop3b_small_TT_bWN1_550_460_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL/TT_bWN1_550_460_Stop3b_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL_combined_BasicMeasurement_model_afterFit.root -o SRsmallRun3.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others,Signal_550_460' -b -y -c SRRun3smallDM0b0,SRRun3smallDM0b1,SRRun3smallDM0b2,SRRun3smallDM1bALL
=============================================================================================================

=============================================================================================================
HistFitter.py -t -w -f -F excl -u "--Year Run3 --Selection Stop3b --region small --Syst" ./ttMET2L_Stop3b.py

HistFitter.py -wtfp -F excl ttMET2L_Stop3b.py -u "--Syst --region small --Year Run3 --Selection Stop3b" -g TT_bWN1_550_460
YieldsTable.py -w results/ttMET2L_Run3_Stop3b_small_TT_bWN1_550_460_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL_exclusion/TT_bWN1_550_460_Stop3b_SRRun3smallDM0b0_SRRun3smallDM0b1_SRRun3smallDM0b2_SRRun3smallDM1bALL_combined_BasicMeasurement_model_afterFit.root -o SRsmallRun3.tex -s 'ttbar,VV,ttZ,Zjets_Sh,Wt_dyn_DR,Others,Signal_550_460' -b -y -c CRTOPsmall1bRun3