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
TOP_MASS, TOP_MASS_WINDOW = 172.5, 70  # GeV
W_MASS, W_MASS_WINDOW = 80, 30  # GeV
FR_PTCUT, SRQQ_PTCUT, SRBQ_PTCUT, FB_PTCUT = 0., 0., 0., 350.  # GeV

# https://registry.hub.docker.com/r/jmduarte/mapyde - docker image for madgraph, pythia8, and delphes

################################



################################
# Baseline function to reconstruct tops
@nb.njit
def reconstruct_top(
    topquarks, bquarks, wbosons, wquarks1, wquarks2, 
    jetfjets, 
    reco_check_func, 
    matched_jetfjets_builder
):
    # Loop over every event
    for topquarks_event, bquarks_event, wbosons_event, wquarks1_event, wquarks2_event, jetfjets_event in zip(
        topquarks, bquarks, wbosons, wquarks1, wquarks2, jetfjets
    ):
        # Loop over every top (+ daughters)
        matched_jetfjets_builder.begin_list()
        for topquark, bquark, wboson, wquark1, wquark2 in zip(
            topquarks_event, bquarks_event, wbosons_event, wquarks1_event, wquarks2_event
        ):  # dont need to check b and w mother index b/c constructed to match
            minDR, minDR_jetfjet_idx = DR_FILL_VALUE, NOJET_FILL_VALUE  # mindeltaR, mindeltaR_jetidx, mindeltaR_fjetidx
            
            # Find the jet(s) and fatjet(s) with the smallest combined deltaR, depending on the reco type
            for i, jetfjet in enumerate(jetfjets_event):
                top_jetfjet_deltaR  = reco_check_func(topquark, bquark, wboson, wquark1, wquark2, jetfjet)
                minDR, minDR_jetfjet_idx = (top_jetfjet_deltaR, i) if top_jetfjet_deltaR < minDR else (minDR, minDR_jetfjet_idx)

            # Add the matched jetfjets to the jetfjet builder
            matched_jetfjets_builder.append(minDR_jetfjet_idx)

        matched_jetfjets_builder.end_list()

    return matched_jetfjets_builder



################################
# Specific reconstruction definitions via deltaR matching
@nb.njit
def FullyBoosted_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of fjet collection with itself, choose 1 (i.e. wraps with 'fjet' field for common treatment downstream)
):
    topquark_deltaR = jetfjet['fjet'].deltaR(topquark)
    bquark_deltaR = jetfjet['fjet'].deltaR(bquark)
    wboson_deltaR = jetfjet['fjet'].deltaR(wboson)
    wquark1_deltaR = jetfjet['fjet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['fjet'].deltaR(wquark2)

    topmass = jetfjet['fjet'].mass
    topmass_window = (topmass > (TOP_MASS - TOP_MASS_WINDOW)) and (topmass < (TOP_MASS + TOP_MASS_WINDOW))
    toppt = jetfjet['fjet'].pt
    toppt_window = toppt > FB_PTCUT


    # Fully-Boosted
    top_jetfjet_deltaR = topquark_deltaR if (
        topquark_deltaR < FJET_DR 
        and bquark_deltaR < FJET_DR 
        and wboson_deltaR < FJET_DR and wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
    ) and topmass_window and toppt_window else DR_FILL_VALUE

    return top_jetfjet_deltaR

@nb.njit
def SemiResolvedQQ_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet and fjet collections
):
    bquark_deltaR = jetfjet['jet'].deltaR(bquark)
    wboson_deltaR = jetfjet['fjet'].deltaR(wboson)
    wquark1_deltaR = jetfjet['fjet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['fjet'].deltaR(wquark2)

    topmass = (jetfjet['fjet'] + jetfjet['jet']).mass
    topmass_window = (topmass > (TOP_MASS - TOP_MASS_WINDOW)) and (topmass < (TOP_MASS + TOP_MASS_WINDOW))
    wmass = jetfjet['fjet'].mass
    wmass_window = (wmass > (W_MASS - W_MASS_WINDOW)) and (wmass < (W_MASS + W_MASS_WINDOW))

    # Semi-Resolved qq
    top_jetfjet_deltaR = (bquark_deltaR + wboson_deltaR) if (
        bquark_deltaR < JET_DR 
        and wboson_deltaR < FJET_DR and wquark1_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
    ) and topmass_window and wmass_window else DR_FILL_VALUE

    return top_jetfjet_deltaR

@nb.njit
def SemiResolvedBQ_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet and fjet collections
):
    topBQ1_jetfjet_deltaR = SemiResolvedBQ1_top(
        topquark, bquark, wboson, wquark1, wquark2, 
        jetfjet
    )
    topBQ2_jetfjet_deltaR = SemiResolvedBQ1_top(
        topquark, bquark, wboson, wquark1, wquark2, 
        jetfjet
    )

    if topBQ2_jetfjet_deltaR < topBQ1_jetfjet_deltaR: return topBQ2_jetfjet_deltaR
    else: return topBQ1_jetfjet_deltaR

@nb.njit
def SemiResolvedBQ1_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet and fjet collections
):
    bquark_deltaR = jetfjet['fjet'].deltaR(bquark)
    wquark1_deltaR = jetfjet['fjet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['jet'].deltaR(wquark2)

    topmass = (jetfjet['fjet'] + jetfjet['jet']).mass
    topmass_window = (topmass > (TOP_MASS - TOP_MASS_WINDOW)) and (topmass < (TOP_MASS + TOP_MASS_WINDOW))

    # Semi-Resolved bq1
    top_jetfjet_deltaR = (bquark_deltaR + wquark1_deltaR + wquark2_deltaR) if (
        bquark_deltaR < FJET_DR and wquark1_deltaR < FJET_DR 
        and wquark2_deltaR < JET_DR
    ) and topmass_window else DR_FILL_VALUE

    return top_jetfjet_deltaR

@nb.njit
def SemiResolvedBQ2_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet and fjet collections
):
    bquark_deltaR = jetfjet['fjet'].deltaR(bquark)
    wquark1_deltaR = jetfjet['jet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['fjet'].deltaR(wquark2)

    topmass = (jetfjet['fjet'] + jetfjet['jet']).mass
    topmass_window = (topmass > (TOP_MASS - TOP_MASS_WINDOW)) and (topmass < (TOP_MASS + TOP_MASS_WINDOW))

    # Semi-Resolved bq2
    top_jetfjet_deltaR = (bquark_deltaR + wquark1_deltaR + wquark2_deltaR) if (
        bquark_deltaR < FJET_DR and wquark2_deltaR < FJET_DR
        and wquark1_deltaR < JET_DR
    ) and topmass_window else DR_FILL_VALUE

    return top_jetfjet_deltaR

@nb.njit
def FullyResolved_top(
    topquark, bquark, wboson, wquark1, wquark2, 
    jetfjet  # Should be the cartesian product of jet collection with itself, choose 3
):
    bquark_deltaR = jetfjet['bjet'].deltaR(bquark)
    wquark1_deltaR = jetfjet['q1jet'].deltaR(wquark1)
    wquark2_deltaR = jetfjet['q2jet'].deltaR(wquark2)

    topmass = (jetfjet['bjet'] + jetfjet['q1jet'] + jetfjet['q2jet']).mass
    topmass_window = (topmass > (TOP_MASS - TOP_MASS_WINDOW)) and (topmass < (TOP_MASS + TOP_MASS_WINDOW))
    wmass = (jetfjet['q1jet'] + jetfjet['q2jet']).mass
    wmass_window = (wmass > (W_MASS - W_MASS_WINDOW)) and (wmass < (W_MASS + W_MASS_WINDOW))

    # Fully-Resolved
    top_jetfjet_deltaR = (bquark_deltaR + wquark1_deltaR + wquark2_deltaR) if (
        bquark_deltaR < JET_DR and wquark1_deltaR < JET_DR and wquark2_deltaR < JET_DR
    ) and topmass_window and wmass_window else DR_FILL_VALUE

    return top_jetfjet_deltaR




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

