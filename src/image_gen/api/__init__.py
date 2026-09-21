"""HTTP wrapper around the same core the CLI uses.

Deployed in three places, which is why it is worth having early:
  1. locally with the stub backend, to exercise the output contract with no GPU
  2. on a cluster compute node, as a warm-model prompt-iteration loop over an
     SSH tunnel -- /docs becomes a prompt form
  3. on the local gateway, serving files to Unity when the gateway and the VR
     PC are different machines
"""
