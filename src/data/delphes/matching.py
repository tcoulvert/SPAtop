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
def match_fjet_to_higgs(tops, bquarks, fjets, builder):
    for tops_event, bquarks_event, fjets_event in zip(tops, bquarks, fjets):
        builder.begin_list()
        for higgs, top_idx in zip(tops_event, tops_event.idx):
            match_idx = -1
            bdaughters = []
            for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
                if bquark_m1 == top_idx:
                    bdaughters.append(bquark)
            for i, fjet in enumerate(fjets_event):
                dr_h = fjet.deltaR(higgs)
                dr_b0 = fjet.deltaR(bdaughters[0])
                dr_b1 = fjet.deltaR(bdaughters[1])
                if dr_h < FJET_DR and dr_b0 < FJET_DR and dr_b1 < FJET_DR:
                    match_idx = i
            builder.append(match_idx)
        builder.end_list()
    return builder


@nb.njit
def match_jets_to_higgs(tops, bquarks, jets, builder):
    for tops_event, bquarks_event, jets_event in zip(tops, bquarks, jets):
        builder.begin_list()
        for _, top_idx in zip(tops_event, tops_event.idx):
            match_idx_b0 = -1
            match_idx_b1 = -1
            bdaughters = []
            for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
                if bquark_m1 == top_idx:
                    bdaughters.append(bquark)
            for i, jet in enumerate(jets_event):
                dr_b0 = jet.deltaR(bdaughters[0])
                dr_b1 = jet.deltaR(bdaughters[1])
                if dr_b0 < JET_DR:
                    match_idx_b0 = i
                if dr_b1 < JET_DR:
                    match_idx_b1 = i
            builder.begin_list()
            builder.append(match_idx_b0)
            builder.append(match_idx_b1)
            builder.end_list()
        builder.end_list()

    return builder


# @nb.njit
# def match_higgs_to_fjet(tops, bquarks, fjets, builder):
#     for tops_event, bquarks_event, fjets_event in zip(tops, bquarks, fjets):
#         builder.begin_list()
#         for i, fjet in enumerate(fjets_event):
#             match_idx = -1
#             for j, (higgs, top_idx) in enumerate(zip(tops_event, tops_event.idx)):
#                 bdaughters = []
#                 for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
#                     if bquark_m1 == top_idx:
#                         bdaughters.append(bquark)
#                 dr_h = fjet.deltaR(higgs)
#                 dr_b0 = fjet.deltaR(bdaughters[0])
#                 dr_b1 = fjet.deltaR(bdaughters[1])
#                 if dr_h < FJET_DR and dr_b0 < FJET_DR and dr_b1 < FJET_DR:
#                     match_idx = j + 1  # index higgs as 1, 2, 3
#             builder.append(match_idx)
#         builder.end_list()
#     return builder


# @nb.njit
# def match_higgs_to_jet(higgs, bquarks, jets, builder):
#     for higgs_event, bquarks_event, jets_event in zip(higgs, bquarks, jets):
#         builder.begin_list()
#         for i, (jet, jet_flv) in enumerate(zip(jets_event, jets_event.flavor)):
#             match_idx = -1
#             for j, (_, higgs_idx) in enumerate(zip(higgs_event, higgs_event.idx)):
#                 for bquark, bquark_m1 in zip(bquarks_event, bquarks_event.m1):
#                     if bquark_m1 == higgs_idx and jet.deltaR(bquark) < JET_DR and np.abs(jet_flv) == 5:
#                         match_idx = j + 1  # index higgs as 1, 2, 3
#             builder.append(match_idx)
#         builder.end_list()

#     return builder

# Do we need this?? # -> for now no. #
# @nb.njit
# def match_wboson_to_jet(wbosons, jets, builder):
#     for wbosons_event, jets_event in zip(wbosons, jets):
#         builder.begin_list()
#         for i, jet in enumerate(jets_event):
#             match_idx = -1
#             for j, wboson in enumerate(wbosons_event):
#                 if jet.deltaR(wboson.d1) < JET_DR or jet.deltaR(wboson.d2) < JET_DR:
#                     match_idx = j + 1  # index wbosons as 1, 2, 3, etc
#             builder.append(match_idx)
#         builder.end_list()

#     return builder


# @nb.njit
# def match_top_to_fjet(
#     tops, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets, 
#     builder, match_type='none'
# ):
#     for tops_event, bquarks_event, wbosons_event, wquarks1_event, wquarks2_event, fjets_event in zip(
#         tops, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets
#     ):
#         builder.begin_list()
#         for i, fjet in enumerate(fjets_event):
#             match_idx = -1
#             for j, (top, top_idx) in enumerate(zip(tops_event, tops_event.idx)):
#                 bdaughter = []
#                 wdaughters = []
#                 for bquark, bquark_m1, wboson_d1, wboson_d2, wboson_m1 in zip(
#                     bquarks_event, bquarks_event.m1, 
#                     wquarks1_event, wquarks2_event, wbosons_event.m1
#                 ):
#                     if bquark_m1 == top_idx:
#                         bdaughter.append(bquark)
#                     if wboson_m1 == top_idx:
#                         wdaughters.extend([wboson_d1, wboson_d2])
#                 dr_t = fjet.deltaR(top)
#                 dr_b = fjet.deltaR(bdaughter[0])
#                 dr_w1 = fjet.deltaR(wdaughters[0])
#                 dr_w2 = fjet.deltaR(wdaughters[1])
#                 if not dr_t < FJET_DR:
#                     continue
#                 if match_type == 'bqq':
#                     if dr_b < FJET_DR and dr_w1 < FJET_DR and dr_w2 < FJET_DR:
#                         match_idx = j + 1  # index top as 1, 2, 3
#                 elif match_type == 'qq':
#                     if not (dr_b < FJET_DR) and dr_w1 < FJET_DR and dr_w2 < FJET_DR:
#                         match_idx = j + 1  # index top as 1, 2, 3
#                 elif match_type == 'bq':
#                     if (
#                         dr_b < FJET_DR and dr_w1 < FJET_DR and not (dr_w2 < FJET_DR)
#                     ) or (
#                         dr_b < FJET_DR and not (dr_w1 < FJET_DR) and dr_w2 < FJET_DR
#                     ):
#                         match_idx = j + 1  # index top as 1, 2, 3
#                 else:
#                     if (
#                         dr_b < FJET_DR and dr_w1 < FJET_DR
#                     ) or (
#                         dr_b < FJET_DR and dr_w2 < FJET_DR
#                     ) or (
#                         dr_w1 < FJET_DR and dr_w2 < FJET_DR
#                     ):
#                         match_idx = j + 1  # index top as 1, 2, 3
#             builder.append(match_idx)
#         builder.end_list()
#     return builder

@nb.njit
def match_top_to_fjet(
    tops, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets, 
    all_fjet_builder, bqq_fjet_builder, bq_fjet_builder, qq_fjet_builder):
    for tops_event, bquarks_event, wbosons_event, wquarks1_event, wquarks2_event, fjets_event in zip(
        tops, bquarks, wbosons, wquarks_d1, wquarks_d2, fjets
    ):
        all_fjet_builder.begin_list()
        bqq_fjet_builder.begin_list()
        bq_fjet_builder.begin_list()
        qq_fjet_builder.begin_list()
        for i, fjet in enumerate(fjets_event):
            all_match_idx, bqq_match_idx, bqq_match_idx, bqq_match_idx = -1, -1, -1, -1
            for j, (top, top_idx) in enumerate(zip(tops_event, tops_event.idx)):
                bdaughter = []
                wdaughters = []
                for bquark, bquark_m1, wboson_d1, wboson_d2, wboson_m1 in zip(
                    bquarks_event, bquarks_event.m1, 
                    wquarks1_event, wquarks2_event, wbosons_event.m1
                ):
                    if bquark_m1 == top_idx:
                        bdaughter.append(bquark)
                    if wboson_m1 == top_idx:
                        wdaughters.extend([wboson_d1, wboson_d2])
                dr_t = fjet.deltaR(top)
                dr_b = fjet.deltaR(bdaughter[0])
                dr_w1 = fjet.deltaR(wdaughters[0])
                dr_w2 = fjet.deltaR(wdaughters[1])

                if not dr_t < FJET_DR:  # top isnt in fatjet, so ignore
                    continue
                bqq_match_bool = (
                    dr_b < FJET_DR and dr_w1 < FJET_DR and dr_w2 < FJET_DR
                )
                bq_match_bool = (
                    dr_b < FJET_DR and dr_w1 < FJET_DR and (not dr_w2 < FJET_DR)
                ) or (
                    dr_b < FJET_DR and (not dr_w1 < FJET_DR) and dr_w2 < FJET_DR
                )
                qq_match_bool = (
                    (not dr_b < FJET_DR) and dr_w1 < FJET_DR and dr_w2 < FJET_DR
                )
                
                if bqq_match_bool or bq_match_bool or qq_match_bool:
                    all_match_idx = j + 1  # index top as 1, 2, 3
                    break
                if bqq_match_bool:
                    bqq_match_idx = j + 1  # index top as 1, 2, 3
                    break
                elif bq_match_bool:
                    bq_match_idx = j + 1  # index top as 1, 2, 3
                    break
                elif qq_match_bool:
                    qq_match_idx = j + 1  # index top as 1, 2, 3
                    break
            all_fjet_builder.append(all_match_idx)
            bqq_fjet_builder.append(bqq_match_idx)
            bq_fjet_builder.append(bq_match_idx)
            qq_fjet_builder.append(qq_match_idx)

        all_fjet_builder.end_list()
        bqq_fjet_builder.end_list()
        bq_fjet_builder.end_list()
        qq_fjet_builder.end_list()

    return all_fjet_builder, bqq_fjet_builder, bq_fjet_builder, qq_fjet_builder

def jet_check(jet, quarks, jet_flav):
    return np.array([
        jet.deltaR(quark) < JET_DR and np.abs(jet_flav) == np.abs(quark.pid) for quark in quarks
    ]), np.array([jet.deltaR(quark) for quark in quarks])

# def jet_deltaR(jet, quarks, jet_fav):
#     return np.array(
#         [jet.deltaR(quark) if jet.deltaR(quark) < JET_DR and np.abs(jet_fav) == np.abs(quark.pid) else 999 for quark in quarks]
#     )

# @nb.njit
def match_top_to_jet(
    tops, bquarks, wbosons, wquarks_d1, wquarks_d2, jets, 
    alljet_builder, bjet_builder, wjet_builder
):
    running_num_diftopjets_overlap = 0
    for bquarks_event, wquarks1_event, wquarks2_event, jets_event in zip(
        bquarks, wquarks_d1, wquarks_d2, jets
    ):
        alljet_builder.begin_list()
        bjet_builder.begin_list()
        wjet_builder.begin_list()
        for i, (jet, jet_flav) in enumerate(zip(jets_event, jets_event.flavor)):
            alljet_match_idx, bjet_match_idx, wjet_match_idx = -1, -1, -1

            bquarks_bool, bquarks_deltaR = [
                jet.deltaR(quark) < JET_DR and np.abs(jet_flav) == np.abs(quark_pid) for quark, quark_pid in zip(bquarks_event, bquarks_event.pid)
            ], [jet.deltaR(quark) for quark in bquarks_event]
            wquarks1_bool, wquarks1_deltaR = [
                jet.deltaR(quark) < JET_DR and np.abs(jet_flav) == np.abs(quark_pid) for quark, quark_pid in zip(wquarks1_event, wquarks1_event.pid)
            ], [jet.deltaR(quark) for quark in wquarks1_event]
            wquarks2_bool, wquarks2_deltaR = [
                jet.deltaR(quark) < JET_DR and np.abs(jet_flav) == np.abs(quark_pid) for quark, quark_pid in zip(wquarks2_event, wquarks2_event.pid)
            ], [jet.deltaR(quark) for quark in wquarks2_event]

            
            # for j, (bquark, wquark1, wquark1_pid, wquark2, wquark2_pid) in enumerate(zip(
            #     bquarks_event,
            #     wquarks1_event, wquarks1_event.pid, 
            #     wquarks2_event, wquarks2_event.pid
            # )):  # dont need to check b and w mother index b/c made them match by construction

            if np.sum(np.logical_or(
                np.logical_or(bquarks_bool, wquarks1_bool), wquarks2_bool
            )) == 1:
                j = np.nonzero(np.logical_or(
                    np.logical_or(bquarks_bool, wquarks1_bool), wquarks2_bool
                ))[0][0]
                alljet_match_idx = j + 1  # index top as 1, 2, 3, etc
            elif np.sum(np.logical_or(
                np.logical_or(bquarks_bool, wquarks1_bool), wquarks2_bool
            )) > 1:
                print("jets from different tops overlapping!!")
                running_num_diftopjets_overlap += 1

                bquarks_filled_deltaR = np.where(bquarks_bool, bquarks_deltaR, 999)
                wquarks1_filled_deltaR = np.where(wquarks1_bool, wquarks1_deltaR, 999)
                wquarks2_filled_deltaR = np.where(wquarks2_bool, wquarks2_deltaR, 999)

                j = np.argmin(np.min(
                    [bquarks_filled_deltaR, wquarks1_filled_deltaR, wquarks2_filled_deltaR],
                    axis=0
                ))
                alljet_match_idx = j + 1  # index top as 1, 2, 3, etc
            else:
                alljet_builder.append(alljet_match_idx)
                bjet_builder.append(bjet_match_idx)
                wjet_builder.append(wjet_match_idx)
                continue

            if bquarks_bool[j] and not (wquarks1_bool[j] or wquarks2_bool[j]):
                bjet_match_idx = j + 1  # index top as 1, 2, 3, etc
            elif not bquarks_bool[j] and (wquarks1_bool[j] or wquarks2_bool[j]):
                wjet_match_idx = j + 1  # index top as 1, 2, 3, etc
            elif bquarks_bool[j] and (wquarks1_bool[j] or wquarks2_bool[j]):
                if bquarks_deltaR[j] < wquarks1_deltaR[j] and bquarks_deltaR[j] < wquarks2_deltaR[j]:
                    bjet_match_idx = j + 1  # index top as 1, 2, 3, etc
                else:
                    wjet_match_idx = j + 1  # index top as 1, 2, 3, etc
            
            alljet_builder.append(alljet_match_idx)
            bjet_builder.append(bjet_match_idx)
            wjet_builder.append(wjet_match_idx)

            # for j, (bquark, wquark1, wquark1_pid, wquark2, wquark2_pid) in enumerate(zip(
            #     bquarks_event,
            #     wquarks1_event, wquarks1_event.pid, 
            #     wquarks2_event, wquarks2_event.pid
            # )):  # dont need to check b and w mother index b/c made them match by construction
                
            #     bquark_match_bool = (
            #         jet.deltaR(bquark) < JET_DR and np.abs(jet_flv) == 5
            #     )  # conditions for bjet match
            #     wquark_match_bool = (
            #         (
            #             jet.deltaR(wquark1) < JET_DR and np.abs(jet_flv) == np.abs(wquark1_pid)
            #         ) or (
            #             jet.deltaR(wquark2) < JET_DR and np.abs(jet_flv) == np.abs(wquark2_pid)
            #         )
            #     )  # conditions for wjet match

            #     if bquark_match_bool or wquark_match_bool:
            #         alljet_match_idx = j + 1  # index top as 1, 2, 3, etc
            #     if bquark_match_bool and not wquark_match_bool:
            #         bjet_match_idx = j + 1  # index top as 1, 2, 3, etc
            #         break
            #     elif not bquark_match_bool and wquark_match_bool:
            #         wjet_match_idx = j + 1  # index top as 1, 2, 3, etc
            #         break
            #     elif bquark_match_bool and wquark_match_bool:
            #         if jet.deltaR(bquark) < jet.deltaR(wquark1) and jet.deltaR(bquark) < jet.deltaR(wquark2):
            #             bjet_match_idx = j + 1  # index top as 1, 2, 3, etc
            #             break
            #         else:
            #             wjet_match_idx = j + 1  # index top as 1, 2, 3, etc
            #             break
                
            #     alljet_builder.append(alljet_match_idx)
            #     bjet_builder.append(bjet_match_idx)
            #     wjet_builder.append(wjet_match_idx)

        alljet_builder.end_list()
        bjet_builder.end_list()
        wjet_builder.end_list()

    print(f"num events where jets from different tops overlap: {running_num_diftopjets_overlap}")
    print(f"fraction of events where jets from different tops overlap: {running_num_diftopjets_overlap / np.shape(bquarks)[0]}")
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
