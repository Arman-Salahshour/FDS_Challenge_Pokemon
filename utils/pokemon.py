from constants import *

def init_unique_game_elements():
    """Initialize the container for unique game elements."""
    return {
        'statuses': set(),
        'effects': set(),
        'move_categories': set(),
        'move_types': set(),
        'move_names': set(),
    }


def update_pokemon_db(pokemon_db: dict, pokemon_entry: dict) -> None:
    """
    Safely update the Pokémon database with a single Pokémon entry.

    pokemon_db structure:
        {
            "Pikachu": {
                "types": [...],
                "base_hp": ...,
                "base_atk": ...,
                ...
            },
            ...
        }
    """
    if not pokemon_entry:
        return

    name = pokemon_entry.get('name')
    if not name or name in pokemon_db:
        return

    pokemon_db[name] = {
        'types': pokemon_entry.get('types', []),
        **{k: v for k, v in pokemon_entry.items() if k.startswith('base_')}
    }


def extract_game_elements_from_timeline(battle_timeline, unique_game_elements: dict) -> None:
    """
    Update unique_game_elements in-place based on a battle timeline.
    """
    if not battle_timeline:
        return

    for turn in battle_timeline:
        # Pokémon state: statuses & effects
        for side in ['p1_pokemon_state', 'p2_pokemon_state']:
            state = turn.get(side)
            if state:
                unique_game_elements['statuses'].add(state.get('status'))
                for effect in state.get('effects', []):
                    unique_game_elements['effects'].add(effect)

        # Move details: categories, types, names
        for side in ['p1_move_details', 'p2_move_details']:
            move = turn.get(side)
            if move:
                unique_game_elements['move_categories'].add(move.get('category'))
                unique_game_elements['move_types'].add(move.get('type'))
                unique_game_elements['move_names'].add(move.get('name'))


def build_knowledge_base(train_data) -> tuple[dict, dict]:
    """
    Single, efficient scan of the training data to build:

    - pokemon_db: mapping from Pokémon name to base stats & types
    - unique_game_elements: sets of statuses, effects, move categories/types/names
    """
    pokemon_db: dict = {}
    unique_game_elements = init_unique_game_elements()

    print("Scanning all training battles to build Knowledge Base...")

    for battle in tqdm(train_data, desc="Mapping Game Universe (Train)"):
        # 1. Learn Pokémon Stats (from P1 team and P2 lead)
        for p1_poke in battle.get('p1_team_details', []):
            update_pokemon_db(pokemon_db, p1_poke)

        p2_lead = battle.get('p2_lead_details')
        if p2_lead:
            update_pokemon_db(pokemon_db, p2_lead)

        # 2. Learn Game Elements (from timeline)
        extract_game_elements_from_timeline(
            battle.get('battle_timeline', []),
            unique_game_elements
        )

    return pokemon_db, unique_game_elements


def print_universe_summary(pokemon_db: dict, unique_game_elements: dict) -> None:
    """Pretty-print the knowledge base summary."""
    print(f"\n--- Universe Mapped (from Training Data) ---")

    # 1. Print the list of unique Pokémon
    pokemon_names = sorted(pokemon_db.keys())
    print(f"\nRegistered {len(pokemon_names)} unique Pokémon species in Knowledge Base:")
    print(f"   {pokemon_names}")

    # 2. Print all unique game elements
    print(f"\nGame Elements Discovered:")
    for category, elements in unique_game_elements.items():
        clean_elements = sorted(str(e) for e in elements if e is not None)
        print(f" - {category.replace('_', ' ').title()} ({len(clean_elements)}):")
        print(f"   {clean_elements}")
