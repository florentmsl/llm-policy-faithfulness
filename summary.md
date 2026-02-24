# Summary

---

## RQ1: Can LLMs detect when symbolic policies are actually working?

**Based on current observations, the answer is: Yes (?).**

### Freeway

- **Full-Context**: The SCoBots policy crosses the road as fast as possible to score. The LLM says it does exactly that.
- **Ungrounded - Only-Policy**: The SCoBots policy advances Obj1 and avoids collisions. The LLM says it does exactly that.

---

## RQ2: Can LLMs detect policy behavior without explicit reward-function access?

**Based on current observations, the answer is: yes (contrary to expectations).**

### Freeway

- The SCoBots policy crosses the road as fast as possible to score. The LLM says it does exactly that.

---

## RQ3: Can LLMs detect misaligned policies (including shortcut learning)?

**Based on current observations, the answer is: Yes.**

### Freeway

- The SCoBots policy always goes up and hits anything until it scores. The LLM finds it.

---

## RQ4: Can LLMs correctly predict how trained symbolic policies adapt under environment simplifications?

**Based on current observations, the answer is: No.**

### Freeway (simplification: all cars stopped)

- The SCoBots policy scores not reliably. The LLM says it does.

---
