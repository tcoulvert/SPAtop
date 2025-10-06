import copy

import awkward as ak
import numba as nb
import numpy as np
import vector

################################


vector.register_awkward()
vector.register_numba()
ak.numba.register_and_check()

################################

JET_DR = 0.5  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L642
FJET_DR = 0.8  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L658
FILL_VALUE = 999

# https://registry.hub.docker.com/r/jmduarte/mapyde - docker image for madgraph, pythia8, and delphes

################################


@nb.njit
def match_top_to_fjet(
    topquarks, bquarks, wbosons, wquarks1, wquarks2, fjets, 
    all_fjet_builder, bqq_fjet_builder, qq_fjet_builder, bq1_fjet_builder, bq2_fjet_builder
):
    qidx_map = {'bqq': 1, 'qq': 2, 'bq1': 3, 'bq2': 3}
    for topquarks_event, bquarks_event, wbosons_event, wquarks1_event, wquarks2_event, fjets_event in zip(
        topquarks, bquarks, wbosons, wquarks1, wquarks2, fjets
    ):
        all_fjet_builder.begin_list()
        bqq_fjet_builder.begin_list()
        qq_fjet_builder.begin_list()
        bq1_fjet_builder.begin_list()
        bq2_fjet_builder.begin_list()
        matched_set = set()
        for i, fjet in enumerate(fjets_event):
            all_match_idx, bqq_match_idx, qq_match_idx, bq1_match_idx, bq2_match_idx = -1, -1, -1, -1, -1

            minDR, minDR_idxs = FILL_VALUE, (FILL_VALUE, '')  # mindeltaR, (mindeltaR_topidx, mindeltaR_quarktype)
            for j, (topquark, bquark, wboson, wquark1, wquark2) in enumerate(zip(
                topquarks_event, bquarks_event, wbosons_event,
                wquarks1_event, wquarks2_event
            )):  # dont need to check b and w mother index b/c made them match by construction
                topquark_deltaR = fjet.deltaR(topquark)
                bquark_deltaR = fjet.deltaR(bquark)
                wboson_deltaR = fjet.deltaR(wboson)
                wquark1_deltaR = fjet.deltaR(wquark1)
                wquark2_deltaR = fjet.deltaR(wquark2)

                # Fully-Boosted
                bqq_deltaR = topquark_deltaR if (
                    topquark_deltaR < FJET_DR and bquark_deltaR < FJET_DR and wboson_deltaR < FJET_DR
                    and wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR 
                    and len({f'b_{j+1}', f'w_{j+1}', f'w1_{j+1}', f'w2_{j+1}'} & matched_set) == 0
                ) else FILL_VALUE
                # Semi-Resolved qq
                qq_deltaR = wboson_deltaR if (
                    wboson_deltaR < FJET_DR and
                    wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
                    and len({f'w_{j+1}', f'w1_{j+1}', f'w2_{j+1}'} & matched_set) == 0
                ) else FILL_VALUE
                # Semi-Resolved bq1
                bq1_avg_deltaR = (
                    bquark_deltaR + wquark1_deltaR
                )/2 if (
                    bquark_deltaR < FJET_DR and wquark1_deltaR < FJET_DR
                    and len({f'b_{j+1}', f'w1_{j+1}'} & matched_set) == 0
                ) else FILL_VALUE
                # Semi-Resolved bq2
                bq2_avg_deltaR = (
                    bquark_deltaR + wquark2_deltaR
                )/2 if (
                    bquark_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
                    and len({f'b_{j+1}', f'w2_{j+1}'} & matched_set) == 0
                ) else FILL_VALUE

                for deltaR, detaR_type in [
                    (bqq_deltaR, 'bqq'), (qq_deltaR, 'qq'), 
                    (bq1_avg_deltaR, 'bq1'), (bq2_avg_deltaR, 'bq2')
                ]:
                    if (
                        deltaR < minDR  # Checks that the deltaR is valid (not FILL_VALUE) and less than the current best...
                        and qidx_map[detaR_type] <= qidx_map[minDR_idxs[1]]  # ...so long as the type follows the priority queue (bqq -> qq -> bq)
                    ):
                        minDR = deltaR
                        minDR_idxs = (j+1, detaR_type)
            
            if minDR != FILL_VALUE:
                all_match_idx = minDR_idxs[0]
                if minDR_idxs[1] == 'bqq':
                    matched_set |= {f'b_{j+1}', f'w_{j+1}', f'w1_{j+1}', f'w2_{j+1}'}
                    bqq_match_idx = minDR_idxs[0]
                elif minDR_idxs[1] == 'qq':
                    matched_set |= {f'w_{j+1}', f'w1_{j+1}', f'w2_{j+1}'}
                    qq_match_idx = minDR_idxs[0]
                elif minDR_idxs[1] == 'bq1':
                    matched_set |= {f'b_{j+1}', f'w1_{j+1}'}
                    bq1_match_idx = minDR_idxs[0]
                elif minDR_idxs[1] == 'bq2':
                    matched_set |= {f'b_{j+1}', f'w2_{j+1}'}
                    bq2_match_idx = minDR_idxs[0]

            all_fjet_builder.append(all_match_idx)
            bqq_fjet_builder.append(bqq_match_idx)
            qq_fjet_builder.append(qq_match_idx)
            bq1_fjet_builder.append(bq1_match_idx)
            bq2_fjet_builder.append(bq2_match_idx)

        all_fjet_builder.end_list()
        bqq_fjet_builder.end_list()
        qq_fjet_builder.end_list()
        bq1_fjet_builder.end_list()
        bq2_fjet_builder.end_list()

    return all_fjet_builder, bqq_fjet_builder, qq_fjet_builder, bq1_fjet_builder, bq2_fjet_builder

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

            mindeltaR, mindeltaR_idxs = FILL_VALUE, (-1, -1)  # mindeltaR, (mindeltaR_topidx, mindeltaR_quarktype)
            for j, (bquark, wquark1, wquark2) in enumerate(zip(
                bquarks_event, wquarks1_event, wquarks2_event
            )):  # dont need to check b and w mother index b/c made them match by construction
                bquark_deltaR = jet.deltaR(bquark) if jet.deltaR(bquark) < JET_DR and jet_btag and f'b_{j+1}' not in matched_set else FILL_VALUE
                wquark1_deltaR = jet.deltaR(wquark1) if jet.deltaR(wquark1) < JET_DR and f'w1_{j+1}' not in matched_set else FILL_VALUE
                wquark2_deltaR = jet.deltaR(wquark2) if jet.deltaR(wquark2) < JET_DR and f'w2_{j+1}' not in matched_set else FILL_VALUE

                wquark_deltaR, wquark_q = (wquark1_deltaR, 1) if wquark1_deltaR < wquark2_deltaR else (wquark2_deltaR, 2)
                min_j_deltaR, q = (bquark_deltaR, 0) if bquark_deltaR < wquark_deltaR else (wquark_deltaR, wquark_q)  # 0 means bquark, 1 or 2 means wquark

                if min_j_deltaR < mindeltaR:
                    mindeltaR = min_j_deltaR
                    mindeltaR_idxs = (j+1, q)  # j+1 b/c index top as 1, 2, 3, etc

            if mindeltaR != FILL_VALUE:
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

