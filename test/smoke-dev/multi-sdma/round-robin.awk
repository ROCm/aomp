# Checks that the SDMA engines used between the "begin/end round-robin" markers
# rotate through a fixed set of engines.

/begin round-robin/ { W = 1; next }
/end round-robin/   { W = 0; next }
/multi-sdma: /      { print; next }
W && sub(/.*SDMA Engine: /, "") {
  E[N++] = V = $0 + 0
  if (!(V in Seen)) { Seen[V]; K++ }
}
END {
  # At least two engines, at least 2 rounds
  OK = K >= 2 && N > 2 * K
  # Check first round in-depth. The "start" of the round might be anywhere, so
  # just check for one wrap-around.
  for (I = 1; I <= K; ++I)
    WrapCount += (E[I % K] <= E[I - 1])
  if (WrapCount != 1)
    OK = 0
  # Check that the other rounds match.
  for (I = K; I < N; ++I)
    if (E[I] != E[I % K])
      OK = 0
  printf "round-robin: %s: %d copies over %d engines\n", OK ? "OK" : "FAIL", N, K
}
