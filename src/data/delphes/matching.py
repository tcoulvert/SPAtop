import copy

import awkward as ak
import numba as nb
import vector

################################


vector.register_awkward()
vector.register_numba()
ak.numba.register_and_check()

################################

JET_DR = 0.5  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L642
FJET_DR = 0.8  # https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl#L658
DR_FILL_VALUE = 999
NOJET_FILL_VALUE = -1

# https://registry.hub.docker.com/r/jmduarte/mapyde - docker image for madgraph, pythia8, and delphes

################################



################################
# Baseline function to reconstruct tops
@nb.njit
def reconstruct_top(
    topquarks, bquarks, wbosons, wquarks1, wquarks2, 
    jetfjets, 
    reco_check_func, 
    matched_jets_builder, matched_fjets_builder
):
    # Loop over every event
    for topquarks_event, bquarks_event, wbosons_event, wquarks1_event, wquarks2_event, jetfjets_event in zip(
        topquarks, bquarks, wbosons, wquarks1, wquarks2, jetfjets
    ):
        # Loop over every top (+ daughters)
        matched_jets_builder.begin_list()
        matched_fjets_builder.begin_list()
        for topquark, bquark, wboson, wquark1, wquark2 in zip(
            topquarks_event, bquarks_event, wbosons_event, wquarks1_event, wquarks2_event
        ):  # dont need to check b and w mother index b/c constructed to match
            _, exmpl_jet_idxs, exmpl_fjet_idxs = reco_check_func(topquark, bquark, wboson, wquark1, wquark2, jetfjets_event[0])
            minDR, minDR_jet_idxs, minDR_fjet_idxs = DR_FILL_VALUE, [NOJET_FILL_VALUE for _ in exmpl_jet_idxs], [NOJET_FILL_VALUE for _ in exmpl_fjet_idxs]  # mindeltaR, mindeltaR_jetidx, mindeltaR_fjetidx
            
            # Find the jet(s) and fatjet(s) with the smallest combined deltaR, depending on the reco type
            for jetfjet in jetfjets_event:
                top_jetfjet_deltaR, top_jet_idxs, top_fjet_idxs = reco_check_func(topquark, bquark, wboson, wquark1, wquark2, jetfjet)
                minDR, minDR_jet_idxs, minDR_fjet_idxs = (top_jetfjet_deltaR, top_jet_idxs, top_fjet_idxs) if top_jetfjet_deltaR < minDR else (minDR, minDR_jet_idxs, minDR_fjet_idxs)

            # Add the matched jets to the jet builder
            matched_jets_builder.begin_list()
            for minDR_jet_idx in minDR_jet_idxs:
                matched_jets_builder.append(minDR_jet_idx)
            matched_jets_builder.end_list()
            # Add the matched fatjets to the fatjet builder
            matched_fjets_builder.begin_list()
            for minDR_fjet_idx in minDR_fjet_idxs:
                matched_fjets_builder.append(minDR_fjet_idx)
            matched_fjets_builder.end_list()

        matched_jets_builder.end_list()
        matched_fjets_builder.end_list()

    return matched_jets_builder, matched_fjets_builder



################################
# Specific reconstruction definitions via deltaR matching
@nb.njit
def FullyBoosted_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be just the fjet collection
):
    topquark_deltaR = jetfjet.deltaR(topquark)
    bquark_deltaR = jetfjet.deltaR(bquark)
    wboson_deltaR = jetfjet.deltaR(wboson)
    wquark1_deltaR = jetfjet.deltaR(wquark1)
    wquark2_deltaR = jetfjet.deltaR(wquark2)

    # Fully-Boosted
    top_jetfjet_deltaR = topquark_deltaR if (
        topquark_deltaR < FJET_DR 
        and bquark_deltaR < FJET_DR 
        and wboson_deltaR < FJET_DR and wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
    ) else DR_FILL_VALUE
    top_jet_idxs, top_fjet_idxs = [NOJET_FILL_VALUE], [jetfjet['idx']]

    return top_jetfjet_deltaR, top_jet_idxs, top_fjet_idxs

@nb.njit
def SemiResolvedQQ_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet and fjet collections
):
    bquark_deltaR = jetfjet['jet'].deltaR(bquark)
    wboson_deltaR = jetfjet['fjet'].deltaR(wboson)
    wquark1_deltaR = jetfjet['fjet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['fjet'].deltaR(wquark2)

    # Semi-Resolved qq
    top_jetfjet_deltaR = (bquark_deltaR + wboson_deltaR) if (
        bquark_deltaR < JET_DR 
        and wboson_deltaR < FJET_DR and wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
    ) else DR_FILL_VALUE
    top_jet_idxs, top_fjet_idxs = [jetfjet['jet']['idx']], [jetfjet['fjet']['idx']]

    return top_jetfjet_deltaR, top_jet_idxs, top_fjet_idxs

@nb.njit
def SemiResolvedBQ1_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet and fjet collections
):
    bquark_deltaR = jetfjet['fjet'].deltaR(bquark)
    wquark1_deltaR = jetfjet['fjet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['jet'].deltaR(wquark2)

    # Semi-Resolved bq1
    top_jetfjet_deltaR = (bquark_deltaR + wquark1_deltaR + wquark2_deltaR) if (
        bquark_deltaR < FJET_DR and wquark1_deltaR < FJET_DR 
        and wquark2_deltaR < JET_DR
    ) else DR_FILL_VALUE
    top_jet_idxs, top_fjet_idxs = [jetfjet['jet']['idx']], [jetfjet['fjet']['idx']]

    return top_jetfjet_deltaR, top_jet_idxs, top_fjet_idxs

@nb.njit
def SemiResolvedBQ2_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet and fjet collections
):
    bquark_deltaR = jetfjet['fjet'].deltaR(bquark)
    wquark1_deltaR = jetfjet['jet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['fjet'].deltaR(wquark2)

    # Semi-Resolved bq2
    top_jetfjet_deltaR = (bquark_deltaR + wquark1_deltaR + wquark2_deltaR) if (
        bquark_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
        and wquark1_deltaR < JET_DR
    ) else DR_FILL_VALUE
    top_jet_idxs, top_fjet_idxs = [jetfjet['jet']['idx']], [jetfjet['fjet']['idx']]

    return top_jetfjet_deltaR, top_jet_idxs, top_fjet_idxs

@nb.njit
def FullyResolved_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet collection with itself, choose 3
):
    bquark_deltaR = jetfjet['jet1'].deltaR(bquark)
    wquark1_deltaR = jetfjet['jet2'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['jet3'].deltaR(wquark2)

    # Fully-Resolved
    top_jetfjet_deltaR = (bquark_deltaR + wquark1_deltaR + wquark2_deltaR) if (
        bquark_deltaR < JET_DR and wquark1_deltaR < JET_DR and wquark2_deltaR < JET_DR
    ) else DR_FILL_VALUE
    top_jet_idxs, top_fjet_idxs = [jetfjet['jet1']['idx'], jetfjet['jet2']['idx'], jetfjet['jet3']['idx']], [NOJET_FILL_VALUE]

    return top_jetfjet_deltaR, top_jet_idxs, top_fjet_idxs




################################
# Matches jets to fatjets for DR mask and attention bias
@nb.njit
def match_fjet_to_jet(fjets, jets, builder, deltaR_builder):
    for fjets_event, jets_event in zip(fjets, jets):
        builder.begin_list()
        deltaR_builder.begin_list()
        for i, jet in enumerate(jets_event):
            minDR, matched_fjet_idx = DR_FILL_VALUE, NOJET_FILL_VALUE
            for j, fjet in enumerate(fjets_event):
                if jet.deltaR(fjet) < FJET_DR:
                    matched_fjet_idx = j
                    minDR = jet.deltaR(fjet)
                elif jet.deltaR(fjet) < minDR:
                    minDR = jet.deltaR(fjet)
            builder.append(matched_fjet_idx)
            deltaR_builder.append(minDR)
        builder.end_list()
        deltaR_builder.end_list()

    return builder, deltaR_builder

