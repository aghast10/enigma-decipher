import itertools, math, random, string
# =======================
#  Enigma M3 (I-V, reflector B/C), modern attack
#  Phases: seed (pos), refine rings, hill-climb plugboard
# =======================

ALPH = string.ascii_uppercase
A2I = {c:i for i,c in enumerate(ALPH)}
I2A = {i:c for c,i in A2I.items()}

ROTORS = {
    "I":   ("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q"),
    "II":  ("AJDKSIRUXBLHWTMCQGZNPYFVOE", "E"),
    "III": ("BDFHJLCPRTXVZNYEIWGAKMUSQO", "V"),
    "IV":  ("ESOVPZJAYQUIRHXLNFTGKDCMWB", "J"),
    "V":   ("VZBRGITYUPSDNHLXAWMJQOFECK", "Z"),
    # If you want to include VI-VIII, add them here
}
REFLECTORS = {
    "B": "YRUHQSLDPXNGOKMIEBFZCWVJAT",
    "C": "FVPJIAOYEDRZXWGCTKUQSBNMHL",
}

def clean_text(s):
    return "".join(ch for ch in s.upper() if ch in A2I)

class Rotor:
    def __init__(self, wiring, notch, ring=0, pos=0):
        self.wiring = [A2I[c] for c in wiring]
        self.inv_wiring = [0]*26
        for i,w in enumerate(self.wiring): self.inv_wiring[w]=i
        self.notch = A2I[notch]
        self.ring  = ring
        self.pos   = pos
    def at_notch(self):
        return self.pos == self.notch
    def step(self):
        self.pos = (self.pos + 1) % 26
    def fwd(self,c):
        x=(c + self.pos - self.ring)%26; y=self.wiring[x]
        return (y - self.pos + self.ring)%26
    def rev(self,c):
        x=(c + self.pos - self.ring)%26; y=self.inv_wiring[x]
        return (y - self.pos + self.ring)%26

class Plugboard:
    def __init__(self, pairs=None):
        self.map = list(range(26))
        if pairs:
            for a,b in pairs:
                ai,bi=A2I[a],A2I[b]
                self.map[ai],self.map[bi]=self.map[bi],self.map[ai]
    def swap(self,c): return self.map[c]
    def pairs(self):
        seen=set(); out=[]
        for a in range(26):
            b=self.map[a]
            if a!=b and (b,a) not in seen:
                out.append((I2A[a], I2A[b])); seen.add((a,b))
        return out

class Enigma:
    def __init__(self, rotor_names=("I","II","III"), rings="AAA", pos="AAA",
                 reflector="B", plug_pairs=None):
        self.right  = Rotor(*ROTORS[rotor_names[2]], ring=A2I[rings[2]], pos=A2I[pos[2]])
        self.middle = Rotor(*ROTORS[rotor_names[1]], ring=A2I[rings[1]], pos=A2I[pos[1]])
        self.left   = Rotor(*ROTORS[rotor_names[0]], ring=A2I[rings[0]], pos=A2I[pos[0]])
        self.ref    = [A2I[c] for c in REFLECTORS[reflector]]
        self.plug   = Plugboard(plug_pairs or [])

    def copy(self):
        e = Enigma()
        e.right  = Rotor("", "A");  e.middle = Rotor("", "A");  e.left = Rotor("", "A")
        for src,dst in [(self.right,e.right),(self.middle,e.middle),(self.left,e.left)]:
            dst.wiring=src.wiring[:]; dst.inv_wiring=src.inv_wiring[:]
            dst.notch=src.notch; dst.ring=src.ring; dst.pos=src.pos
        e.ref=self.ref[:]; e.plug=Plugboard(self.plug.pairs())
        return e

    def step_rotors(self):
        # double-stepping
        if self.middle.at_notch():
            self.left.step(); self.middle.step()
        elif self.right.at_notch():
            self.middle.step()
        self.right.step()

    def enc_letter(self, ch):
        if ch not in A2I: return ch
        self.step_rotors()
        c=A2I[ch]
        c=self.plug.swap(c)
        c=self.right.fwd(c); c=self.middle.fwd(c); c=self.left.fwd(c)
        c=self.ref[c]
        c=self.left.rev(c); c=self.middle.rev(c); c=self.right.rev(c)
        c=self.plug.swap(c)
        return I2A[c]

    def enc(self, text):
        text=clean_text(text)
        return "".join(self.enc_letter(ch) for ch in text)

# -------- Scoring (simple German: bigrams/trigrams + frequent words) --------
GER_BIGRAMS = "ER EN CH DE EI TE IN NG ND SC ST RE GE BE UN HE AN AU SE DI IE NE IS".split()
GER_TRIGRAMS = "DER DIE UND ICH NIC EIN SCH CHE DEN GEN UNS ZUR MIT END EIT BER".split()
GER_WORDS = "DER DIE DAS UND IST NICHT EIN EINE MIT VON ZUR FUR WETTER BERICHT NORDSEE BEFEHL UHR ANGRIFF SEKTOR MITTERNACHT UFER SICHERN UBOOT FEIND KONVOI".split()

def score_text(pt):
    # Combine counts of bigrams, trigrams and words; log-saturated
    s = 0.0
    # bigrams
    for bg in GER_BIGRAMS:
        s += pt.count(bg)
    # trigrams
    for tg in GER_TRIGRAMS:
        s += 2.0 * pt.count(tg)
    # words (bonus)
    for w in GER_WORDS:
        c = pt.count(w)
        if c: s += 4.0 * min(c, 3)
    # penalty for suspicious repeated characters (anti-language)
    # (very mild: Enigma did not produce 'J' in German military use with J->I, but we keep it neutral)
    return s

# -------- Utilities --------
def all_positions(step=1):
    letters = ALPH[::step] if step>1 else ALPH
    for a in letters:
        for b in letters:
            for c in letters:
                yield a+b+c

def all_rings():
    for r in itertools.product(ALPH, repeat=3):
        yield "".join(r)

def best_k(items, k, key=lambda x:x):
    # keep top-k simple
    out=[]
    for it in items:
        out.append(it); out.sort(key=key, reverse=True)
        if len(out)>k: out.pop()
    return out

# -------- Phase 1 search: seed without plugboard, rings=AAA --------
def seed_search(ciphertext, rotor_set=("I","II","III","IV","V"),
                reflectors=("B","C"), pos_step=1,
                sample_len=200, keep_top=50):
    ct = clean_text(ciphertext)
    ct_sample = ct[:sample_len]
    seeds=[]
    for ro in itertools.permutations(rotor_set, 3):
        for rf in reflectors:
            for pos in all_positions(step=pos_step):
                e = Enigma(rotor_names=ro, rings="AAA", pos=pos, reflector=rf, plug_pairs=[])
                pt = e.enc(ct_sample)
                sc = score_text(pt)
                seeds = best_k(seeds + [(sc, ro, "AAA", pos, rf)], keep_top, key=lambda x:x[0])
    return seeds  # list of (score, rotors, rings, pos, refl)

# -------- Phase 2 search: refine rings for best seeds --------
def refine_rings_for_seed(ct, seed, sample_len=240, keep_top=5):
    sc0, ro, _, pos, rf = seed
    ct_sample = clean_text(ct)[:sample_len]
    best=[]
    for rings in all_rings():
        e = Enigma(rotor_names=ro, rings=rings, pos=pos, reflector=rf, plug_pairs=[])
        pt = e.enc(ct_sample)
        sc = score_text(pt)
        best = best_k(best + [(sc, ro, rings, pos, rf)], keep_top, key=lambda x:x[0])
    return best

# -------- Phase 3: Hill-climbing plugboard (up to N pairs) --------
def hillclimb_plugboard(ct, ro, rings, pos, rf, max_pairs=10,
                        iters=8000, start_temp=3.0, sample_len=None, seed_pairs=None, rnd=None):
    rnd = rnd or random.Random(0xC0FFEE)
    ct = clean_text(ct)
    if sample_len:
        ct_sample = ct[:sample_len]
    else:
        ct_sample = ct
    # current state
    pairs = list(seed_pairs or [])
    best_pairs = list(pairs)

    def decrypt_score(pairs_list):
        e = Enigma(rotor_names=ro, rings=rings, pos=pos, reflector=rf, plug_pairs=pairs_list)
        pt = e.enc(ct_sample)
        return score_text(pt), pt

    best_score, best_pt = decrypt_score(pairs)
    cur_score = best_score

    alphabet = list(ALPH)
    def random_move(cur_pairs):
        cur = set(tuple(sorted(p)) for p in cur_pairs)
        # three move types: add/remove/swap a pair
        move_type = rnd.choice(["add","swap","remove"])
        new = [tuple(p) for p in cur_pairs]
        if move_type=="add" and len(new) < max_pairs:
            free = [a for a in alphabet if all(a not in p for p in new)]
            if len(free)>=2:
                a,b = rnd.sample(free,2)
                new.append(tuple(sorted((a,b))))
        elif move_type=="remove" and len(new)>0:
            idx = rnd.randrange(len(new))
            new.pop(idx)
        else:  # swap: re-pair four letters if at least 2 pairs exist
            if len(new)>=2:
                i,j = rnd.sample(range(len(new)), 2)
                a,b = new[i]
                c,d = new[j]
                # produce (a,c) and (b,d) with prob 0.5 or (a,d),(b,c)
                if rnd.random()<0.5:
                    cand = [p for k,p in enumerate(new) if k not in (i,j)] + [tuple(sorted((a,c))), tuple(sorted((b,d)))]
                else:
                    cand = [p for k,p in enumerate(new) if k not in (i,j)] + [tuple(sorted((a,d))), tuple(sorted((b,c)))]
                new = cand
        # normalize pairs ordered and without duplicates
        norm = []
        used=set()
        for a,b in new:
            if a in used or b in used or a==b: continue
            used.add(a); used.add(b)
            norm.append(tuple(sorted((a,b))))
        norm.sort()
        return norm

    T = start_temp
    for it in range(iters):
        cand_pairs = random_move(best_pairs if rnd.random()<0.3 else pairs)
        cand_score, _ = decrypt_score(cand_pairs)
        accept = cand_score > cur_score or rnd.random() < math.exp( (cand_score-cur_score) / max(1e-6,T) )
        if accept:
            pairs = cand_pairs
            cur_score = cand_score
            if cur_score > best_score:
                best_score, best_pairs = cur_score, pairs[:]
                best_score_pt = Enigma(rotor_names=ro, rings=rings, pos=pos, reflector=rf, plug_pairs=best_pairs).enc(ct)
        # cooling schedule
        if (it+1) % 200 == 0:
            T *= 0.92
    # final output on full text
    final_pt = Enigma(rotor_names=ro, rings=rings, pos=pos, reflector=rf, plug_pairs=best_pairs).enc(ct)
    return {
        "score": best_score,
        "rotors": ro,
        "rings": rings,
        "pos": pos,
        "reflector": rf,
        "plugboard": best_pairs,
        "plaintext": final_pt
    }

# -------- Full pipeline --------
def break_enigma_like_allies(ciphertext,
                             rotor_set=("I","II","III","IV","V"),
                             reflectors=("B","C"),
                             pos_step=1, seed_keep=60, seed_sample_len=200,
                             rings_keep=6, rings_sample_len=240,
                             hill_iters=10000, hill_pairs=10, hill_sample_len=None,
                             random_seed=12345,
                             crib=None):
    rnd = random.Random(random_seed)
    ct = clean_text(ciphertext)

    # (Optional) filter by CRIB to speed up Phase 1 (if provided)
    def contains_crib(pt):
        return (crib is None) or (clean_text(crib) in pt)

    # Phase 1: seeds without plugboard, rings=AAA
    seeds = seed_search(ct, rotor_set=rotor_set, reflectors=reflectors,
                        pos_step=pos_step, sample_len=seed_sample_len, keep_top=seed_keep)
    if crib:
        seeds = [s for s in seeds if contains_crib(Enigma(rotor_names=s[1], rings="AAA", pos=s[3], reflector=s[4]).enc(ct[:seed_sample_len]))] or seeds

    # Phase 2: refine rings for best seeds
    ring_candidates=[]
    for s in seeds[:rings_keep]:
        ring_candidates += refine_rings_for_seed(ct, s, sample_len=rings_sample_len, keep_top=1)
    ring_candidates.sort(key=lambda x:x[0], reverse=True)
    # Best candidate so far:
    _, ro, rings, pos, rf = ring_candidates[0]

    # Phase 3: hill-climb plugboard
    result = hillclimb_plugboard(ct, ro, rings, pos, rf,
                                 max_pairs=hill_pairs, iters=hill_iters,
                                 start_temp=3.5, sample_len=hill_sample_len, rnd=rnd)

    return result

# =================== Quick Example ===================
enigma_snd = Enigma(rotor_names=("I","II","III"), rings="AAA", pos="AAA",
                 reflector="B", plug_pairs=None)
enigma_rcv = Enigma(rotor_names=("I","II","III"), rings="AAA", pos="AAA",
                 reflector="B", plug_pairs=None)

if __name__ == "__main__":
    PLAINTEXT = "WETTERBERICHT VON DER NORDSEE UHR MITTERNACHT SICHERN UFER UBOOT KONVOI ANGRIFF"
    CIPH = enigma_snd.enc(PLAINTEXT) # works better with long phrases. Only simple German.
    print("PLAINTEXT: ", enigma_rcv.enc(CIPH))  # should be the plaintext
    print("CIPHERTEXT:", CIPH)
    out = break_enigma_like_allies(
        CIPH,
        rotor_set=("I","II","III","IV","V"),
        reflectors=("B","C"),
        pos_step=2,          # step >1 accelerates (2 or 3) at the cost of some accuracy
        seed_keep=50,
        seed_sample_len=100, # use 150-250 if your PC can handle it
        rings_keep=6,
        rings_sample_len=120,
        hill_iters=12000,    # increase to 50k-100k for top quality
        hill_pairs=10,
        hill_sample_len=None,
        random_seed=1337,
        crib='WETTER'            # if you know a crib (e.g. "WETTER"), put it here
    )
    print("ROTORS:", out["rotors"], "RINGS:", out["rings"], "POS:", out["pos"], "REFLECTOR:", out["reflector"])
    print("PLUGBOARD:", " ".join(a+b for a,b in out["plugboard"]))
    print("DECIPHERED:", out["plaintext"])