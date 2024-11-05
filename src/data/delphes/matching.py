import copy

import awkward as ak
import numba as nb
import numpy as np
import vector

vector.register_awkward()
vector.register_numba()
ak.numba.register_and_check()

JET_DR = 0.5  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L642
FJET_DR = 0.8  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L658


@nb.njit
def match_top_to_fjet(
    bquarks, wquarks1, wquarks2, fjets, 
    all_fjet_builder, bqq_fjet_builder, bq_fjet_builder, qq_fjet_builder
):
    for bquarks_event, wquarks1_event, wquarks2_event, fjets_event in zip(
        bquarks, wquarks1, wquarks2, fjets
    ):
        all_fjet_builder.begin_list()
        bqq_fjet_builder.begin_list()
        bq_fjet_builder.begin_list()
        qq_fjet_builder.begin_list()
        matched_set = set()
        for i, fjet in enumerate(fjets_event):
            all_match_idx, bqq_match_idx, bq_match_idx, qq_match_idx = -1, -1, -1, -1

            mindeltaR, mindeltaR_idxs = 999, (-1, -1)  # mindeltaR, (mindeltaR_topidx, mindeltaR_quarktype)
            for j, (bquark, wquark1, wquark2) in enumerate(zip(
                bquarks_event,
                wquarks1_event, wquarks2_event
            )):  # dont need to check b and w mother index b/c made them match by construction
                bquark_deltaR = fjet.deltaR(bquark)
                wquark1_deltaR = fjet.deltaR(wquark1)
                wquark2_deltaR = fjet.deltaR(wquark2)

                bqq_avg_deltaR = (
                    bquark_deltaR + wquark1_deltaR + wquark2_deltaR
                )/3 if (
                    bquark_deltaR < FJET_DR and 
                    wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR and len({f'b_{j+1}', f'w1_{j+1}', f'w2_{j+1}'} & matched_set) == 0
                ) else 999
                if bqq_avg_deltaR < mindeltaR or (bqq_avg_deltaR < 999 and mindeltaR_idxs[1] != 0):  # 0 means bqq fatjet
                    mindeltaR = bqq_avg_deltaR
                    mindeltaR_idxs = (j+1, 0)  # j+1 b/c index top as 1, 2, 3, etc

                bq1_avg_deltaR = (
                    bquark_deltaR + wquark1_deltaR
                )/2 if (
                    bquark_deltaR < FJET_DR and wquark1_deltaR < FJET_DR
                    and len({f'b_{j+1}', f'w1_{j+1}'} & matched_set) == 0
                ) else 999
                bq2_avg_deltaR = (
                    bquark_deltaR + wquark2_deltaR
                )/2 if (
                    bquark_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
                    and len({f'b_{j+1}', f'w2_{j+1}'} & matched_set) == 0
                ) else 999
                qq_avg_deltaR = (
                    wquark1_deltaR + wquark2_deltaR
                )/2 if (
                    wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
                    and len({f'w1_{j+1}', f'w2_{j+1}'} & matched_set) == 0
                ) else 999

                bq_avg_deltaR, bq_avg_q = (bq1_avg_deltaR, 1) if bq1_avg_deltaR < bq2_avg_deltaR else (bq2_avg_deltaR, 2)
                min_j_avg_deltaR, q = (bq_avg_deltaR, bq_avg_q) if bq_avg_deltaR < qq_avg_deltaR else (qq_avg_deltaR, 3)  # 1 or 2 means bq fatjet, 3 means qq fatjet
                
                if min_j_avg_deltaR < mindeltaR and mindeltaR_idxs[1] != 1:
                    mindeltaR = min_j_avg_deltaR
                    mindeltaR_idxs = (j+1, q)  # j+1 b/c index top as 1, 2, 3, etc
            
            if mindeltaR != 999:
                if mindeltaR_idxs[1] <= 2:
                    matched_set.add(f'b_{j+1}') 
                if mindeltaR_idxs[1] <= 1 or mindeltaR_idxs[1] == 3:
                    matched_set.add(f'w1_{j+1}')
                if mindeltaR_idxs[1] == 0 or mindeltaR_idxs[1] >= 2:
                    matched_set.add(f'w2_{j+1}')
                all_match_idx = mindeltaR_idxs[0]
                bqq_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 0 else bqq_match_idx
                bq_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 1 or mindeltaR_idxs[1] == 2 else bq_match_idx
                qq_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 3 else qq_match_idx

            all_fjet_builder.append(all_match_idx)
            bqq_fjet_builder.append(bqq_match_idx)
            bq_fjet_builder.append(bq_match_idx)
            qq_fjet_builder.append(qq_match_idx)

        all_fjet_builder.end_list()
        bqq_fjet_builder.end_list()
        bq_fjet_builder.end_list()
        qq_fjet_builder.end_list()

    return all_fjet_builder, bqq_fjet_builder, bq_fjet_builder, qq_fjet_builder

# @nb.njit
# def match_top_to_jet(
#     quark_fUniqueID, jet_ref_arrays,
#     jet_builder
# ):
#     for quark_event, jet_event in zip(quark_fUniqueID, jet_ref_arrays):
#         jet_builder.begin_list()

#         for jet_ref_array in jet_event:
#             match_flag, match_idx = False, -1
            
#             for top_idx, quark_ID in enumerate(quark_event):
#                 # if match_flag:
#                 #     break
#                 # print(quark_ID)
#                 # print(jet_ref_array)
                
#                 # for jet_ref in jet_ref_array:
#                 #     if jet_ref == quark_ID:
#                 #         match_idx = top_idx + 1  # +1 b/c index top as 1, 2, 3, etc
#                 #         match_flag = True
#                 #         break
#                 match_idx = top_idx if quark_ID in set(jet_ref_array) else -1
            
#             jet_builder.append(match_idx)
        
#         jet_builder.end_list()

#     return jet_builder

# @nb.njit
# def match_top_to_jet(
#     bquarks, wquarks1, wquarks2, jets, 
#     alljet_builder, bjet_builder, wjet_builder
# ):
#     for bquarks_event, wquarks1_event, wquarks2_event, jets_event in zip(
#         bquarks, wquarks1, wquarks2, jets
#     ):
#         alljet_builder.begin_list()
#         bjet_builder.begin_list()
#         wjet_builder.begin_list()

#         # matched_jet_set = {}
#         # for i, bquark in enumerate(bquarks_event):
#         #     bquark_deltaR = []
#         bquark_deltaRs, w1quark_deltaRs, w2quark_deltaRs = [], [], []
#         for jet_idx, (jet, jet_flav) in enumerate(zip(jets_event, jets_event.flavor)):  # Is jet_flav a useful quantity?
#             bquark_deltaRs.append(jet.deltaR(bquarks_event))
#             w1quark_deltaRs.append(jet.deltaR(wquarks1_event))
#             w2quark_deltaRs.append(jet.deltaR(wquarks2_event))
        
#         matched_jet_set = {}
#         for deltaRs in [bquark_deltaRs, w1quark_deltaRs, w2quark_deltaRs]:

#             min1_deltaR, min1_idx, min2_deltaR, min2_idx = 999, -1, 999, -1
#             for jet_idx, deltaR in enumerate(deltaRs):
#                 for q, quark_dR in enumerate(deltaR):

#                 if (
#                     deltaR[0] < min1_deltaR and jet_idx not in matched_jet_set
#                 ):
#                     min1_deltaR = deltaR[0]
#                     min1_idx = jet_idx
#                 if (
#                     deltaR[1] < min2_deltaR and jet_idx not in matched_jet_set
#                 ) and (
#                     jet_idx != min1_idx or deltaR[1] < deltaR[0]
#                 ):
#                     min1_deltaR = deltaR[0]
#                     min1_idx = jet_idx
#                 min1_deltaR = deltaR[0]  else min1_deltaR
#                 min2_deltaR = deltaR[1] if deltaR[1] < min2_deltaR and jet_idx not in matched_jet_set else min2_deltaR
#             min_deltaR = np.argmin(deltaRs, axis=0)


#             if mindeltaR != 999:
#                 alljet_match_idx = mindeltaR_idxs[0]
#                 bjet_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 1 else bjet_match_idx
#                 wjet_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 2 else wjet_match_idx

#             alljet_builder.append(alljet_match_idx)
#             bjet_builder.append(bjet_match_idx)
#             wjet_builder.append(wjet_match_idx)

#         alljet_builder.end_list()
#         bjet_builder.end_list()
#         wjet_builder.end_list()

#     return alljet_builder, bjet_builder, wjet_builder

@nb.njit
def match_top_to_jet(
    bquarks, wquarks1, wquarks2, jets, 
    alljet_builder, bjet_builder, wjet_builder
):
    for bquarks_event, wquarks1_event, wquarks2_event, jets_event in zip(
        bquarks, wquarks1, wquarks2, jets
    ):
        alljet_builder.begin_list()
        bjet_builder.begin_list()
        wjet_builder.begin_list()
        matched_set = set()
        for i, (jet, jet_flav) in enumerate(zip(jets_event, jets_event.flavor)):
            alljet_match_idx, bjet_match_idx, wjet_match_idx = -1, -1, -1

            mindeltaR, mindeltaR_idxs = 999, (-1, -1)  # mindeltaR, (mindeltaR_topidx, mindeltaR_quarktype)
            for j, (bquark, wquark1, wquark1_pid, wquark2, wquark2_pid) in enumerate(zip(
                bquarks_event,
                wquarks1_event, wquarks1_event.pid, 
                wquarks2_event, wquarks2_event.pid
            )):  # dont need to check b and w mother index b/c made them match by construction
                # bquark_deltaR = jet.deltaR(bquark) if jet.deltaR(bquark) < JET_DR and f'b_{j+1}' not in matched_set and np.abs(jet_flav) == 5 else 999
                # wquark1_deltaR = jet.deltaR(wquark1) if jet.deltaR(wquark1) < JET_DR and f'w1_{j+1}' not in matched_set and np.abs(jet_flav) == np.abs(wquark1_pid) else 999
                # wquark2_deltaR = jet.deltaR(wquark2) if jet.deltaR(wquark2) < JET_DR and f'w2_{j+1}' not in matched_set and np.abs(jet_flav) == np.abs(wquark2_pid) else 999
                bquark_deltaR = jet.deltaR(bquark) if jet.deltaR(bquark) < JET_DR and f'b_{j+1}' not in matched_set else 999
                wquark1_deltaR = jet.deltaR(wquark1) if jet.deltaR(wquark1) < JET_DR and f'w1_{j+1}' not in matched_set else 999
                wquark2_deltaR = jet.deltaR(wquark2) if jet.deltaR(wquark2) < JET_DR and f'w2_{j+1}' not in matched_set else 999

                wquark_deltaR, wquark_q = (wquark1_deltaR, 1) if wquark1_deltaR < wquark2_deltaR else (wquark2_deltaR, 2)
                min_j_deltaR, q = (bquark_deltaR, 0) if bquark_deltaR < wquark_deltaR else (wquark_deltaR, wquark_q)  # 0 means bquark, 1 or 2 means wquark

                if min_j_deltaR < mindeltaR:
                    mindeltaR = min_j_deltaR
                    mindeltaR_idxs = (j+1, q)  # j+1 b/c index top as 1, 2, 3, etc

            if mindeltaR != 999:
                matched_set.add(f"{'b' if mindeltaR_idxs[1] == 0 else ('w1' if mindeltaR_idxs[1] == 1 else 'w2')}_{mindeltaR_idxs[0]}")
                alljet_match_idx = mindeltaR_idxs[0]
                bjet_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 0 else bjet_match_idx
                wjet_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 1 or mindeltaR_idxs[1] == 2 else wjet_match_idx

            alljet_builder.append(alljet_match_idx)
            bjet_builder.append(bjet_match_idx)
            wjet_builder.append(wjet_match_idx)

        alljet_builder.end_list()
        bjet_builder.end_list()
        wjet_builder.end_list()

    return alljet_builder, bjet_builder, wjet_builder

@nb.njit
def match_fjet_to_jet(fjets, jets, builder):
    for fjets_event, jets_event in zip(fjets, jets):
        builder.begin_list()
        for i, jet in enumerate(jets_event):
            match_idx = -1
            for j, fjet in enumerate(fjets_event):
                if jet.deltaR(fjet) < FJET_DR:
                    match_idx = j
            builder.append(match_idx)
        builder.end_list()

    return builder
