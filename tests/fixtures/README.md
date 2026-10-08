`tttt_delphes_24.parquet`: 24 events of all-hadronic pp -> tttt Delphes output
(MadGraph + Pythia8 + Delphes, CMS card), entries 0, 5, 7, 17, 24, 32, 36, 50,
62, 65, 72, 76, 77, 91, 97, 101, 103, 106, 108, 117, 125, 129, 136, 138 of
`tttt_hadronic_0.root` from the 2026-08 15M-event production. Only the 36
branches `convert_to_h5.py` reads are kept, as awkward arrays, zstd-19.
Events were chosen so that each passes event selection with four hadronic
tops, which maximises the number of targets per byte.
