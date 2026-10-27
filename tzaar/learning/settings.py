"""The settings of training. config.py can change them for one model (models/<name>/config.json)."""

MIN_SHARE_TO_PRINT = 0.005   # opponents less likely than this are left out of the log line
STEPS_PER_GENERATION = 3      # self-play, training, testing
RANDOM_GAMES = 1500
START_GAMES = 100             # games per self-play batch and per test match in generation 0...
GAMES_PER_GENERATION = 20     # ...growing by this many games for each generation after

# New training games the best net plays each generation. More fresh games gave better nets (the jumps at 859, 903
# and 937 came with more data and fewer epochs). A position is about 3.2 KB, so a 5000-game batch is roughly
# 670 MB; the limit is memory (12 GB machine), which is why BATCHES_TO_KEEP is small. Lower this if it runs out.
HIDDEN_NEURONS = 256          # neurons in the first hidden layer. A saved net with fewer is widened to this when training resumes (it plays
                              # exactly as before); a net with more is left as it is. Older nets stay narrow and still serve as opponents.
HIDDEN2_NEURONS = 64          # neurons in the second hidden layer, widened the same way (new neurons start switched off)
MAX_SELF_PLAY_GAMES = 5000
MAX_MATCH_GAMES = 600         # games per test match: enough that the score is accurate to about +-2 points
WARM_START = True             # new nets start as a copy of the best net instead of from scratch

# How many passes (epochs) a new net makes over its training data. Each position is labeled with who won the whole
# game, which is a noisy signal. A weak net still has a lot to learn, so more passes help. A strong net mostly
# fits the noise with extra passes, and it drifts away from the best net it started from (its challengers kept
# scoring below 0.5 against the best net at 10 and 6 epochs, and jumped when this went down to 2 and then 1).
WARM_START_EPOCHS_AT_ZERO = 6   # passes per new net when the best net is rated 0...
WARM_START_ELO_PER_EPOCH = 100  # ...one fewer for every this many Elo, down to a minimum of 1
WARM_START_LEARNING_RATE = 0.02
OPPONENTS_BACK = 3            # a new net is rated against this many of the latest accepted nets
SELF_PLAY_EPSILON = 0.2       # share of random moves in self-play, for variety
SELF_PLAY_SEARCH_TURNS = 0    # turns of lookahead for the best net in self-play (0 = none, it just plays by the net)...
SELF_PLAY_SEARCH_KEEP = 3     # ...trying this many moves at each step. Older nets it plays against never search.
SEARCH_LABEL_SHARE = 0        # share of recorded positions labeled by the lookahead's value instead of only the game result (0 = none)...
SEARCH_LABEL_WEIGHT = 0.5     # ...label = (1 - this) * game result + this * the lookahead's value
SEARCH_LABEL_TURNS = 2        # the lookahead used for those labels (turns ahead)...
SEARCH_LABEL_KEEP = 3         # ...and moves tried at each step
SELF_PLAY_OPPONENTS_SEARCH = False   # True: the nets it plays against look ahead too (same turns and keep), not just the best net
SPARRING_PARTNER = ""         # a net to train against: a model's name (its best net, models/<name>/model.npz) or a path ending in .npz
                              # (e.g. "baselines/evo1_gen0432.npz", which never changes). "" = none. Its games are recorded like any others.
SPARRING_SHARE = 0.5          # share of self-play games played against the partner
SPARRING_PARTNER_TURNS = 0    # turns of lookahead the partner uses in those games (0 = it plays plainly)...
SPARRING_PARTNER_KEEP = 3     # ...trying this many moves at each step
SPARRING_TEST_GAMES = 200     # games each newly accepted net plays against the partner, plain vs plain, to see how far it has come (0 = none)
BASELINE_SHARE = 0            # share of self-play games played against the nets in baselines/ (0 = none), split evenly between them
OLDER_NET_WEIGHT = 0.5        # in self-play, each net one step older is this much less likely to be the opponent

# How many recent batches of games a new net trains on. Games played by an older best net no longer match how the
# current one plays, and after each accepted net the challengers dipped until the old batches rolled out of the
# window, so a short window of big fresh batches keeps the data current. (Memory is the other limit, see above.)
BATCHES_TO_KEEP = 2

# A new net is accepted only if all of these hold:
MIN_SCORE_VS_BEST = 0.52      # 1. it wins at least this share of games against the best net
MIN_RATING_GAIN = 5           # 2. its rating (head-to-head vs the last few accepted nets, random left out)

                              #    is at least this many Elo above the best net's
MIN_SCORE_VS_RANDOM = 0.90    # 3. safety floor: it still beats a random player this often
NEAR_MISS_SCORE = 0.50        # a rejected net that still won this much vs the best is kept as a self-play opponent

# The weights for the newer inputs (SIGNALS_START on) are only kept when a net is accepted, so a best net that
# was made before those inputs existed has zeros there, and every challenger would have to learn them from
# scratch and then lose them again when rejected. So each time a rejected challenger scores higher against the
# best net than any donor so far, it becomes a "donor" (the latest few are kept). New challengers get the average
# of the donors' weights for those inputs, which smooths out lucky scores. The donors are dropped when a net is
# accepted, because the new best net carries its own weights for them.
# The donors also supply the parts of the net that were added later and so haven't been proven yet: the hidden
# neurons beyond the first PROVEN_HIDDEN (the net used to be that narrow) and the second hidden layer.
DONORS_TO_KEEP = 3            # averaging a few smooths out donors that only scored well by luck (tests are +-2 points)
RECENT_DONOR_WINDOW = 0       # 0 = off. If set (e.g. 5), one of the DONORS_TO_KEEP donor slots is a rolling one: the best rejected net of the
                              # last this-many generations, even if it scored below the lasting donors. Not saved, so a restart begins without it.

# Alternating plain and smart phases (see phases.py). 0 = off. Otherwise a smart phase gives up after this many generations
# without an accepted net. The smart phase is the normal settings with these overrides on top, e.g.
# {"SELF_PLAY_SEARCH_TURNS": 2, "SELF_PLAY_OPPONENTS_SEARCH": true} or {"SEARCH_LABEL_SHARE": 0.3}. Not saved: a restart begins in the plain phase.
SMART_PHASE_GENERATIONS = 0
SMART_PHASE_SETTINGS = {}
CURRENT_PHASE = ""            # "plain" or "smart" while phases are on; set by the training, shown on the dashboard...
CURRENT_PHASE_NOTE = ""       # ...with this (how far a smart phase is from its cap)
PROVEN_HIDDEN = 64            # the first neurons: well tuned, always from the best net, so a donor can't unsettle them

# A net that fails narrowly gets one re-test against the best net before it is rejected, because a
# normal test (a few hundred games) is only accurate to about +-2.5 points. The re-test result alone decides:
# MIN_SCORE_VS_BEST accepts it, NEAR_MISS_SCORE keeps it as an opponent.
RETEST_GAMES = 1600
RETEST_MIN_SCORE = 0.48       # "borderline" 1: rating clears the bar and it won at least this much vs the best net
RETEST_RATING_SLACK = 20      # "borderline" 2: it won enough vs the best net but its rating is short by up to this much
