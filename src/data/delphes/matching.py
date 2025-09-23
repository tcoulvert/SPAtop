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
VFJET_DR = 0.8

# https://registry.hub.docker.com/r/jmduarte/mapyde - docker image for madgraph, pythia8, and delphes


@nb.njit
def match_top_to_vfjet(
    topquarks, bquarks, wquarks1, wquarks2, vfjets, 
    all_vfjet_builder, bqq_vfjet_builder,
):
    for topquarks_event, bquarks_event, wquarks1_event, wquarks2_event, vfjets_event in zip(
        topquarks, bquarks, wquarks1, wquarks2, vfjets
    ):
        all_vfjet_builder.begin_list()
        bqq_vfjet_builder.begin_list()
        matched_set = set()
        for i, vfjet in enumerate(vfjets_event):
            all_match_idx, bqq_match_idx = -1, -1

            mindeltaR, mindeltaR_idxs = 999, (-1, -1)  # mindeltaR, (mindeltaR_topidx, mindeltaR_quarktype)
            for j, (topquark, bquark, wquark1, wquark2) in enumerate(zip(
                topquarks_event, bquarks_event,
                wquarks1_event, wquarks2_event
            )):  # dont need to check b and w mother index b/c made them match by construction
                topquark_deltaR = vfjet.deltaR(topquark)
                bquark_deltaR = vfjet.deltaR(bquark)
                wquark1_deltaR = vfjet.deltaR(wquark1)
                wquark2_deltaR = vfjet.deltaR(wquark2)

                bqq_avg_deltaR = (
                    topquark_deltaR + bquark_deltaR + wquark1_deltaR + wquark2_deltaR
                )/4 if (
                    topquark_deltaR < VFJET_DR and bquark_deltaR < VFJET_DR and
                    wquark1_deltaR < VFJET_DR and wquark2_deltaR < VFJET_DR and len({f'b_{j+1}', f'w1_{j+1}', f'w2_{j+1}'} & matched_set) == 0
                ) else 999
                if bqq_avg_deltaR < mindeltaR:
                    mindeltaR = bqq_avg_deltaR
                    mindeltaR_idxs = (j+1, 0)  # j+1 b/c index top as 1, 2, 3, etc

            if mindeltaR != 999:
                matched_set.add(f'b_{j+1}')
                matched_set.add(f'w1_{j+1}')
                matched_set.add(f'w2_{j+1}')
                all_match_idx = mindeltaR_idxs[0]
                bqq_match_idx = mindeltaR_idxs[0]

            all_vfjet_builder.append(all_match_idx)
            bqq_vfjet_builder.append(bqq_match_idx)

        all_vfjet_builder.end_list()
        bqq_vfjet_builder.end_list()

    return all_vfjet_builder, bqq_vfjet_builder

@nb.njit
def match_top_to_fjet(
    bquarks, wquarks1, wquarks2, fjets, 
    all_fjet_builder, bq1_fjet_builder, bq2_fjet_builder, qq_fjet_builder
):
    for bquarks_event, wquarks1_event, wquarks2_event, fjets_event in zip(
        bquarks, wquarks1, wquarks2, fjets
    ):
        all_fjet_builder.begin_list()
        bq1_fjet_builder.begin_list()
        bq2_fjet_builder.begin_list()
        qq_fjet_builder.begin_list()
        matched_set = set()
        for i, fjet in enumerate(fjets_event):
            all_match_idx, bq1_match_idx, bq2_match_idx, qq_match_idx = -1, -1, -1, -1

            mindeltaR, mindeltaR_idxs = 999, (-1, -1)  # mindeltaR, (mindeltaR_topidx, mindeltaR_quarktype)
            for j, (bquark, wquark1, wquark2) in enumerate(zip(
                bquarks_event,
                wquarks1_event, wquarks2_event
            )):  # dont need to check b and w mother index b/c made them match by construction
                bquark_deltaR = fjet.deltaR(bquark)
                wquark1_deltaR = fjet.deltaR(wquark1)
                wquark2_deltaR = fjet.deltaR(wquark2)

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
                if mindeltaR_idxs[1] == 1 or mindeltaR_idxs[1] == 2:
                    matched_set.add(f'b_{j+1}') 
                if mindeltaR_idxs[1] == 1 or mindeltaR_idxs[1] == 3:
                    matched_set.add(f'w1_{j+1}')
                if mindeltaR_idxs[1] == 2 or mindeltaR_idxs[1] == 3:
                    matched_set.add(f'w2_{j+1}')
                all_match_idx = mindeltaR_idxs[0]
                bq1_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 1 else bq1_match_idx
                bq2_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 2 else bq2_match_idx
                qq_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 3 else qq_match_idx

            all_fjet_builder.append(all_match_idx)
            bq1_fjet_builder.append(bq1_match_idx)
            bq2_fjet_builder.append(bq2_match_idx)
            qq_fjet_builder.append(qq_match_idx)

        all_fjet_builder.end_list()
        bq1_fjet_builder.end_list()
        bq2_fjet_builder.end_list()
        qq_fjet_builder.end_list()

    return all_fjet_builder, bq1_fjet_builder, bq2_fjet_builder, qq_fjet_builder

@nb.njit
def match_top_to_jet(
    bquarks, wquarks1, wquarks2, jets, 
    alljet_builder, bjet_builder, wjet1_builder, wjet2_builder
):
    for bquarks_event, wquarks1_event, wquarks2_event, jets_event in zip(
        bquarks, wquarks1, wquarks2, jets
    ):
        alljet_builder.begin_list()
        bjet_builder.begin_list()
        wjet1_builder.begin_list()
        wjet2_builder.begin_list()
        matched_set = set()
        for i, (jet, jet_btag) in enumerate(zip(jets_event, jets_event.btag)):
            alljet_match_idx, bjet_match_idx, wjet1_match_idx, wjet2_match_idx = -1, -1, -1, -1

            mindeltaR, mindeltaR_idxs = 999, (-1, -1)  # mindeltaR, (mindeltaR_topidx, mindeltaR_quarktype)
            for j, (bquark, wquark1, wquark2) in enumerate(zip(
                bquarks_event, wquarks1_event, wquarks2_event
            )):  # dont need to check b and w mother index b/c made them match by construction
                bquark_deltaR = jet.deltaR(bquark) if jet.deltaR(bquark) < JET_DR and jet_btag and f'b_{j+1}' not in matched_set else 999
                # bquark_deltaR = jet.deltaR(bquark) if jet.deltaR(bquark) < JET_DR and f'b_{j+1}' not in matched_set else 999
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
                wjet1_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 1 else wjet1_match_idx
                wjet2_match_idx = mindeltaR_idxs[0] if mindeltaR_idxs[1] == 2 else wjet2_match_idx

            alljet_builder.append(alljet_match_idx)
            bjet_builder.append(bjet_match_idx)
            wjet1_builder.append(wjet1_match_idx)
            wjet2_builder.append(wjet2_match_idx)

        alljet_builder.end_list()
        bjet_builder.end_list()
        wjet1_builder.end_list()
        wjet2_builder.end_list()

    return alljet_builder, bjet_builder, wjet1_builder, wjet2_builder


@nb.njit
def match_fjet_to_jet(fjets, jets, builder, deltaR_builder):
    for fjets_event, jets_event in zip(fjets, jets):
        builder.begin_list()
        deltaR_builder.begin_list()
        for i, jet in enumerate(jets_event):
            match_idx, default_DR = -1, 5
            for j, fjet in enumerate(fjets_event):
                if jet.deltaR(fjet) < FJET_DR:
                    match_idx = j
                    default_DR = jet.deltaR(fjet)
                elif jet.deltaR(fjet) < default_DR:
                    default_DR = jet.deltaR(fjet)
            builder.append(match_idx)
            deltaR_builder.append(default_DR)
        builder.end_list()
        deltaR_builder.end_list()

    return builder, deltaR_builder

