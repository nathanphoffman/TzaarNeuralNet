"""The settings of training. config.py can change them for one model (models/<name>/config.json)."""

MIN_SHARE_TO_PRINT = 0.005   # opponents less likely than this are left out of the log line
STEPS_PER_GENERATION = 3      # self-play, training, testing
RANDOM_GAMES = 1500
START_GAMES = 100             # games per self-play batch and per test match in generation 0...
GAMES_PER_GENERATION = 20     # ...growing by this many games for each generation after

# New training games the best net plays each generation. More fresh games gave better nets (the jumps at 859, 903
# and 937 came with more data and fewer epochs). A position is about 3.2 KB, so a 5000-game batch is roughly
# 670 MB; the limit is memory (12 GB machine), which is why BATCHES_TO_KEEP is small. Lower this if it runs out.
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
PROVEN_HIDDEN = 64            # the first neurons: well tuned, always from the best net, so a donor can't unsettle them

# A net that fails narrowly gets one re-test against the best net before it is rejected, because a
# normal test (a few hundred games) is only accurate to about +-2.5 points. The re-test result alone decides:
# MIN_SCORE_VS_BEST accepts it, NEAR_MISS_SCORE keeps it as an opponent.
RETEST_GAMES = 1600
RETEST_MIN_SCORE = 0.48       # "borderline" 1: rating clears the bar and it won at least this much vs the best net
RETEST_RATING_SLACK = 20      # "borderline" 2: it won enough vs the best net but its rating is short by up to this much
