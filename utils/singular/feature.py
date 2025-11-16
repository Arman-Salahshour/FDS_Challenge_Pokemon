from constants import *
import pandas as pd


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

BASE_STATS: List[str] = [
    "base_hp",
    "base_atk",
    "base_def",
    "base_spa",
    "base_spd",
    "base_spe",
]


# -------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------

def _get_stats_from_db(name: str, p_db: Dict[str, Any]) -> Dict[str, float]:
    """
    Internal helper to safely get stats from the pokemon_db.

    Parameters
    ----------
    name : str
        Pokémon name.
    p_db : dict
        Pokemon database: {name: {... base_*, types, ...}, ...}

    Returns
    -------
    dict
        Mapping of each BASE_STAT to float.
    """
    stats = {stat: 0.0 for stat in BASE_STATS}
    db_entry = p_db.get(name)
    if db_entry:
        for stat in BASE_STATS:
            stat_val = db_entry.get(stat, 0.0)
            stats[stat] = float(stat_val) if isinstance(stat_val, (int, float)) else 0.0
    return stats


def _compute_team_mean_stats(
    team_details: List[Dict[str, Any]],
    p_db: Dict[str, Any],
    prefix: str,
) -> Dict[str, float]:
    """
    Compute mean base stats for a team and return as prefixed features.
    """
    team_stats_list = []
    for pokemon in team_details:
        name = pokemon.get("name")
        if name:
            team_stats_list.append(_get_stats_from_db(name, p_db))

    features: Dict[str, float] = {}

    if team_stats_list:
        df = pd.DataFrame(team_stats_list)
        for stat in BASE_STATS:
            features[f"{prefix}_mean_{stat}"] = float(df[stat].mean())
    else:
        for stat in BASE_STATS:
            features[f"{prefix}_mean_{stat}"] = 0.0

    return features


def _get_turn_30_snapshot(
    battle: Dict[str, Any],
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Internal helper to process the timeline and find the final state
    of all known Pokémon over the considered turns.

    Historically used as "Turn 30 snapshot", but you can pass
    max_turns to limit how many timeline entries are processed.

    Parameters
    ----------
    battle : dict
        Single battle record.
    max_turns : int or None
        If provided, only the first `max_turns` entries of the
        timeline are processed. If None, the full timeline is used.

    Returns
    -------
    dict
        {
            'p1_final_states': {name: state, ...},
            'p2_final_states': {name: state, ...},
            'p2_seen_count': int
        }
    """
    timeline = battle.get("battle_timeline", [])
    if max_turns is not None:
        timeline = timeline[:max_turns]

    # 1. Initialize P1's "state board" from the full 6-pokemon team
    p1_team_names = {
        p.get("name") for p in battle.get("p1_team_details", []) if p.get("name")
    }
    p1_last_state: Dict[str, Dict[str, Any]] = {
        name: {"hp_pct": 1.0, "status": "nostatus", "boosts": {}} for name in p1_team_names
    }

    # 2. Initialize P2's "state board" starting with only the lead
    p2_last_state: Dict[str, Dict[str, Any]] = {}
    p2_seen_names = set()

    p2_lead = battle.get("p2_lead_details")
    if p2_lead and p2_lead.get("name"):
        p2_lead_name = p2_lead["name"]
        p2_seen_names.add(p2_lead_name)
        p2_last_state[p2_lead_name] = {
            "hp_pct": 1.0,
            "status": "nostatus",
            "boosts": {},
        }

    # 3. Process the timeline, updating the "state boards"
    for turn_state in timeline:
        # P1 state update
        p1_state = turn_state.get("p1_pokemon_state")
        if p1_state and p1_state.get("name") in p1_last_state:
            p1_last_state[p1_state["name"]] = p1_state  # overwrite with latest

        # P2 state update (and dynamic discovery)
        p2_state = turn_state.get("p2_pokemon_state")
        if p2_state and p2_state.get("name"):
            p2_name = p2_state["name"]
            p2_seen_names.add(p2_name)
            p2_last_state[p2_name] = p2_state  # overwrite with latest

    # 4. Return the final states and the count of seen P2 Pokémon
    return {
        "p1_final_states": p1_last_state,
        "p2_final_states": p2_last_state,
        "p2_seen_count": len(p2_seen_names),
    }


# -------------------------------------------------------------------
# Public feature extraction functions
# -------------------------------------------------------------------

def extract_static_features_v1(
    battle: Dict[str, Any],
    p_db: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract Snapshot 1 (Turn 0) features.

    - P1: Mean stats of all 6 team members.
    - P2: Stats of the single lead Pokémon.
    """
    features: Dict[str, float] = {}

    # P1 Mean Stats (Full 6-Pokémon Team)
    features.update(
        _compute_team_mean_stats(
            battle.get("p1_team_details", []),
            p_db,
            prefix="p1",
        )
    )

    # P2 Lead Stats (Only the revealed lead)
    p2_lead = battle.get("p2_lead_details")
    if p2_lead and p2_lead.get("name"):
        p2_lead_stats = _get_stats_from_db(p2_lead["name"], p_db)
        for stat in BASE_STATS:
            features[f"p2_lead_{stat}"] = p2_lead_stats[stat]
    else:
        for stat in BASE_STATS:
            features[f"p2_lead_{stat}"] = 0.0

    return features


def extract_dynamic_features_v2(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract Snapshot 2 (Turn 30) features:
    - Fainted counts
    - HP levels
    - P2 unseen count (and corresponding HP assumption).
    """
    features: Dict[str, float] = {}
    p1_states = snapshot["p1_final_states"]
    p2_states = snapshot["p2_final_states"]

    # 1. Fainted Counts
    p1_fainted = sum(1 for state in p1_states.values() if state.get("status") == "fnt")
    p2_fainted = sum(1 for state in p2_states.values() if state.get("status") == "fnt")

    features["p1_fainted_count"] = float(p1_fainted)
    features["p2_fainted_count"] = float(p2_fainted)
    features["fainted_advantage"] = float(p2_fainted - p1_fainted)  # higher is better for P1

    # 2. HP Levels (Sum of HP% for non-fainted Pokémon)
    p1_total_hp = sum(
        state.get("hp_pct", 0.0) for state in p1_states.values() if state.get("status") != "fnt"
    )
    p2_total_hp = sum(
        state.get("hp_pct", 0.0) for state in p2_states.values() if state.get("status") != "fnt"
    )

    # 3. P2 Unseen Pokémon
    p2_seen_count = snapshot["p2_seen_count"]
    p2_unseen_count = max(0, 6 - p2_seen_count)  # Max 6 Pokemon per team

    # Assume unseen Pokémon have 100% HP (1.0)
    p2_total_hp_estimated = p2_total_hp + p2_unseen_count

    features["p1_total_hp_pct"] = float(p1_total_hp)
    features["p2_total_hp_pct_estimated"] = float(p2_total_hp_estimated)
    features["hp_advantage"] = float(p1_total_hp - p2_total_hp_estimated)
    features["p2_unseen_count"] = float(p2_unseen_count)

    return features


def extract_other_dynamic_features_v3(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract Iteration 3 (REFINED) features from the Turn 30 snapshot:

    - Granular status conditions (incapacitated, major debuff, passive damage)
    - Separated stat boosts (offensive, defensive, speed)
    """
    features: Dict[str, float] = {}
    p1_states = snapshot["p1_final_states"]
    p2_states = snapshot["p2_final_states"]

    # === 1. REFINED Status Conditions ===
    INCAPACITATED = {"slp", "frz"}
    MAJOR_DEBUFF = {"par", "brn"}
    PASSIVE_DAMAGE = {"psn", "tox"}

    status_groups = {
        "incapacitated": INCAPACITATED,
        "major_debuff": MAJOR_DEBUFF,
        "passive_damage": PASSIVE_DAMAGE,
    }

    # Initialize counts
    for side_prefix in ["p1", "p2"]:
        for group_name in status_groups:
            features[f"{side_prefix}_{group_name}_count"] = 0.0

    # Count P1 statuses
    for state in p1_states.values():
        if state.get("status") != "fnt":
            for group_name, status_set in status_groups.items():
                if state.get("status") in status_set:
                    features[f"p1_{group_name}_count"] += 1.0

    # Count P2 statuses
    for state in p2_states.values():
        if state.get("status") != "fnt":
            for group_name, status_set in status_groups.items():
                if state.get("status") in status_set:
                    features[f"p2_{group_name}_count"] += 1.0

    # === 2. REFINED Stat Boosts ===
    p1_boosts = {"off": 0.0, "def": 0.0, "spe": 0.0}
    p2_boosts = {"off": 0.0, "def": 0.0, "spe": 0.0}

    # Sum P1 boosts
    for state in p1_states.values():
        if state.get("status") != "fnt":
            boost_dict = state.get("boosts", {})
            p1_boosts["off"] += float(boost_dict.get("atk", 0) + boost_dict.get("spa", 0))
            p1_boosts["def"] += float(boost_dict.get("def", 0) + boost_dict.get("spd", 0))
            p1_boosts["spe"] += float(boost_dict.get("spe", 0))

    # Sum P2 boosts
    for state in p2_states.values():
        if state.get("status") != "fnt":
            boost_dict = state.get("boosts", {})
            p2_boosts["off"] += float(boost_dict.get("atk", 0) + boost_dict.get("spa", 0))
            p2_boosts["def"] += float(boost_dict.get("def", 0) + boost_dict.get("spd", 0))
            p2_boosts["spe"] += float(boost_dict.get("spe", 0))

    # Create final boost features
    features["p1_offensive_boost_sum"] = p1_boosts["off"]
    features["p2_offensive_boost_sum"] = p2_boosts["off"]
    features["offensive_boost_advantage"] = p1_boosts["off"] - p2_boosts["off"]

    features["p1_defensive_boost_sum"] = p1_boosts["def"]
    features["p2_defensive_boost_sum"] = p2_boosts["def"]
    features["defensive_boost_advantage"] = p1_boosts["def"] - p2_boosts["def"]

    features["p1_speed_boost_sum"] = p1_boosts["spe"]
    features["p2_speed_boost_sum"] = p2_boosts["spe"]
    features["speed_boost_advantage"] = p1_boosts["spe"] - p2_boosts["spe"]

    return features


# -------------------------------------------------------------------
# High-level convenience API
# -------------------------------------------------------------------

def build_turn_30_snapshot(battle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public wrapper around the internal snapshot builder, for the common
    "Turn 30 snapshot" use case.
    """
    # If your timeline is full, you can pass max_turns=30 here.
    # If you're already truncating before calling this, leave as is.
    return _get_turn_30_snapshot(battle, max_turns=None)


def extract_all_features_v1_v3(
    battle: Dict[str, Any],
    p_db: Dict[str, Any],
) -> Dict[str, float]:
    """
    Convenience function: extract all v1 + v2 + v3 features for a battle.

    Returns
    -------
    dict
        Flattened feature dict suitable for ML models.
    """
    features: Dict[str, float] = {}

    # Static (team & lead)
    features.update(extract_static_features_v1(battle, p_db))

    # Dynamic snapshots (Turn 30-like)
    snapshot = build_turn_30_snapshot(battle)
    features.update(extract_dynamic_features_v2(snapshot))
    features.update(extract_other_dynamic_features_v3(snapshot))

    return features



def create_features(data: List[Dict[str, Any]],
                    p_db: Dict[str, Any],
                    iteration_name: str) -> (pd.DataFrame, pd.Series):
    """
    Main pipeline to create a feature DataFrame based on the iteration name.

    - v1: Static features only.
    - v2: v1 + Dynamic HP/Fainted features.
    - v3: v2 + Dynamic Status/Boost features.
    """
    print(f"\n--- Creating features for '{iteration_name}' --- ")
    feature_list = []
    labels = []

    if not p_db:
        raise ValueError("Pokemon DB is empty! Please re-run Section 2.3.")

    for battle in tqdm(data, desc=f"Processing '{iteration_name}' features"):
        features = {}

        # --- Snapshot 1 (Turn 0) ---
        if iteration_name in ['v1', 'v2', 'v3']:
            features.update(extract_static_features_v1(battle, p_db))

        # --- Snapshot 2 (Turn 30) ---
        if iteration_name in ['v2', 'v3']:
            # Get the final state of the battle ONCE
            snapshot_30 = _get_turn_30_snapshot(battle)
            # Add v2 features (HP, Fainted)
            features.update(extract_dynamic_features_v2(snapshot_30))

        # --- Refinements (Other Dynamics) ---
        if iteration_name in ['v3']:
            # We re-use the snapshot_30 object created in the v2 block
            features.update(extract_other_dynamic_features_v3(snapshot_30))

        # --- Add Metadata ---
        features[BATTLE_ID] = battle.get(BATTLE_ID)
        if TARGET in battle:
            labels.append(int(battle[TARGET]))

        feature_list.append(features)

    # Create DataFrame
    X = pd.DataFrame(feature_list).fillna(0)
    y = pd.Series(labels, name=TARGET) if labels else None

    print(f"Successfully created feature set with {X.shape[1] - 1} features.")

    # Separate features from metadata
    feature_cols = [col for col in X.columns if col not in [TARGET, BATTLE_ID]]

    return X[feature_cols], y

# -------------------------------------------------------------------
# Optional: module-level demo / sanity check
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("Feature extraction module loaded.")
    print("Available functions: extract_static_features_v1, extract_dynamic_features_v2,")
    print("extract_other_dynamic_features_v3, build_turn_30_snapshot, extract_all_features_v1_v3.")
