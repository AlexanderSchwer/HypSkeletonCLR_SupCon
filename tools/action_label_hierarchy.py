from collections import OrderedDict


NTU60_CLASS_NAMES_SOURCE = "https://rose1.ntu.edu.sg/dataset/actionRecognition/"

NTU60_CLASS_NAMES = OrderedDict([
    (0, "drink water"),
    (1, "eat meal"),
    (2, "brush teeth"),
    (3, "brush hair"),
    (4, "drop"),
    (5, "pick up"),
    (6, "throw"),
    (7, "sit down"),
    (8, "stand up"),
    (9, "clapping"),
    (10, "reading"),
    (11, "writing"),
    (12, "tear up paper"),
    (13, "put on jacket"),
    (14, "take off jacket"),
    (15, "put on a shoe"),
    (16, "take off a shoe"),
    (17, "put on glasses"),
    (18, "take off glasses"),
    (19, "put on a hat/cap"),
    (20, "take off a hat/cap"),
    (21, "cheer up"),
    (22, "hand waving"),
    (23, "kicking something"),
    (24, "reach into pocket"),
    (25, "hopping"),
    (26, "jump up"),
    (27, "phone call"),
    (28, "play with phone/tablet"),
    (29, "type on a keyboard"),
    (30, "point to something"),
    (31, "taking a selfie"),
    (32, "check time (from watch)"),
    (33, "rub two hands"),
    (34, "nod head/bow"),
    (35, "shake head"),
    (36, "wipe face"),
    (37, "salute"),
    (38, "put palms together"),
    (39, "cross hands in front"),
    (40, "sneeze/cough"),
    (41, "staggering"),
    (42, "falling down"),
    (43, "headache"),
    (44, "chest pain"),
    (45, "back pain"),
    (46, "neck pain"),
    (47, "nausea/vomiting"),
    (48, "fan self"),
    (49, "punch/slap"),
    (50, "kicking"),
    (51, "pushing"),
    (52, "pat on back"),
    (53, "point finger"),
    (54, "hugging"),
    (55, "giving object"),
    (56, "touch pocket"),
    (57, "shaking hands"),
    (58, "walking towards"),
    (59, "walking apart"),
])


def _node(name, children=None, class_id=None, metadata=None):
    return {
        "name": str(name),
        "children": list(children or []),
        "class_id": class_id,
        "metadata": dict(metadata or {}),
    }


def _leaf(class_id):
    return _node(NTU60_CLASS_NAMES[class_id], class_id=int(class_id))


HYPSKELETONCLR_WARD_TREE = _node("Root", [
    _node("personal_and_mundane_tasks", [
        _node("head_worn_items", [
            _node("put_on_head_items", [_leaf(17), _leaf(19)]),
            _node("take_off_head_items", [_leaf(18), _leaf(20)]),
        ]),
        _node("health_face_and_head", [
            _node("illness_symptoms", [_leaf(40), _leaf(47)]),
            _node("face_head_care", [
                _leaf(3),
                _leaf(36),
                _node("head_neck_pain", [_leaf(43), _leaf(46)]),
            ]),
        ]),
        _node("positive_upper_body_gestures", [_leaf(21), _leaf(22)]),
        _node("paper_phone_and_typing", [
            _leaf(48),
            _node("clap_and_rub_hands", [_leaf(9), _leaf(33)]),
            _leaf(12),
            _leaf(28),
            _node("desk_reading_writing", [_leaf(10), _leaf(11), _leaf(29)]),
        ]),
        _node("pain_drop_and_reach", [
            _node("torso_back_pain", [_leaf(44), _leaf(45)]),
            _node("drop_and_pocket", [_leaf(4), _leaf(24)]),
        ]),
        _node("phone_food_and_hygiene", [
            _leaf(37),
            _leaf(27),
            _node("food_and_drink", [_leaf(1), _leaf(0), _leaf(2)]),
        ]),
        _node("hand_signs_attention_and_selfie", [
            _node("hand_signs", [_leaf(38), _leaf(39)]),
            _node("attention_and_selfie", [_leaf(32), _leaf(30), _leaf(31)]),
        ]),
    ]),
    _node("body_and_interaction_actions", [
        _node("social_and_interactive_actions", [
            _node("object_and_hand_contact", [_leaf(55), _leaf(57), _leaf(56)]),
            _node("directed_other_person_actions", [
                _leaf(52),
                _leaf(53),
                _node("physical_contact_actions", [_leaf(6), _leaf(54), _leaf(50), _leaf(49), _leaf(51)]),
            ]),
        ]),
        _node("full_body_movements", [
            _node("dynamic_and_locomotion", [
                _leaf(23),
                _node("instability_and_hop", [_leaf(41), _leaf(25)]),
                _leaf(35),
                _node("walking_pair", [_leaf(58), _leaf(59)]),
            ]),
            _node("posture_jump_and_jacket", [
                _node("stand_jump", [_leaf(8), _leaf(26)]),
                _node("jacket_actions", [_leaf(13), _leaf(14)]),
                _node("sit_fall", [_leaf(7), _leaf(42)]),
            ]),
            _node("bending_and_lower_body", [
                _node("shoe_actions", [_leaf(15), _leaf(16)]),
                _node("pickup_and_bow", [_leaf(5), _leaf(34)]),
            ]),
        ]),
    ]),
])


STORED_HIERARCHIES = OrderedDict([
    (
        "hypskeletonclr_ward",
        {
            "name": "hypskeletonclr_ward",
            "tree": HYPSKELETONCLR_WARD_TREE,
            "label_space": "ntu60_zero_based",
            "source": "HypSkeletonCLR supervised contrastive loss training, Ward clustering of NTU-60 class prototypes.",
            "default_group_nodes": [
                "personal_and_mundane_tasks",
                "social_and_interactive_actions",
                "full_body_movements",
            ],
        },
    ),
])

def list_hierarchies():
    return list(STORED_HIERARCHIES.keys())


def get_hierarchy(name):
    key = _resolve_hierarchy_name(name)
    return STORED_HIERARCHIES[key]


def hierarchy_class_names(name):
    names = OrderedDict()
    for leaf in _iter_leaves(get_hierarchy(name)["tree"]):
        class_id = leaf.get("class_id")
        if class_id is not None:
            names[int(class_id)] = leaf["name"]
    return names


def hierarchy_leaf_ids(name):
    ids = []
    for leaf in _iter_leaves(get_hierarchy(name)["tree"]):
        class_id = leaf.get("class_id")
        if class_id is not None:
            ids.append(int(class_id))
    return ids


def hierarchy_leaf_paths(name, identifier="class_id"):
    paths = OrderedDict()
    _collect_leaf_paths(get_hierarchy(name)["tree"], [], paths, identifier)
    return paths


def hierarchy_leaf_groups(name, group_node_names=None, identifier="class_id"):
    record = get_hierarchy(name)
    if group_node_names is None:
        group_node_names = record.get("default_group_nodes") or []
    groups = OrderedDict()
    for group_name in group_node_names:
        node = _find_node(record["tree"], group_name)
        if node is None:
            raise ValueError(f"Hierarchy '{record['name']}' has no node '{group_name}'.")
        values = _leaf_values(node, identifier)
        if values:
            groups[str(group_name)] = values
    return groups


def hierarchy_to_nested_dict(name):
    return _public_node(get_hierarchy(name)["tree"])


def _resolve_hierarchy_name(name):
    if not name:
        raise ValueError("Hierarchy name must be non-empty.")
    key = str(name).strip()
    if key not in STORED_HIERARCHIES:
        raise ValueError(
            "Unknown hierarchy '{}'. Available: {}.".format(
                name,
                ", ".join(list_hierarchies()),
            )
        )
    return key


def _find_node(node, name):
    if node["name"] == name:
        return node
    for child in node["children"]:
        found = _find_node(child, name)
        if found is not None:
            return found
    return None


def _iter_leaves(node):
    if not node["children"]:
        yield node
        return
    for child in node["children"]:
        yield from _iter_leaves(child)


def _leaf_values(node, identifier):
    values = []
    for leaf in _iter_leaves(node):
        value = _leaf_identifier(leaf, identifier)
        if value is not None:
            values.append(value)
    return values


def _leaf_identifier(leaf, identifier):
    if identifier == "class_id":
        class_id = leaf.get("class_id")
        return None if class_id is None else int(class_id)
    if identifier == "name":
        return leaf["name"]
    raise ValueError("identifier must be 'class_id' or 'name'.")


def _collect_leaf_paths(node, ancestors, paths, identifier):
    path = ancestors + [node["name"]]
    if not node["children"]:
        key = _leaf_identifier(node, identifier)
        if key is not None:
            paths[key] = path
        return
    for child in node["children"]:
        _collect_leaf_paths(child, path, paths, identifier)


def _public_node(node):
    public = OrderedDict()
    public["name"] = node["name"]
    if node.get("class_id") is not None:
        public["class_id"] = int(node["class_id"])
    if node.get("metadata"):
        public["metadata"] = dict(node["metadata"])
    if node["children"]:
        public["children"] = [_public_node(child) for child in node["children"]]
    return public
