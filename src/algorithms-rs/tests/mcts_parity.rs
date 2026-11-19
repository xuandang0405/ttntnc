use algorithms_rs::alphazero::mcts::{backup, expand, select_puct, Arena};

#[test]
fn expansion_preserves_action_order_and_priors() {
    let (mut arena, root) = Arena::with_root(1.0);
    expand(&mut arena, root, &[17, 4, 9], &[0.2, 0.5, 0.3]).unwrap();
    let edges = arena.get(root).children.as_ref().unwrap();
    assert_eq!(edges.iter().map(|e| e.action_idx).collect::<Vec<_>>(), vec![17, 4, 9]);
    assert!((arena.get(edges[1].child).prior - 0.5).abs() < 1e-12);
}

#[test]
fn puct_uses_opponent_view_and_first_edge_for_ties() {
    let (mut arena, root) = Arena::with_root(1.0);
    expand(&mut arena, root, &[3, 8], &[0.5, 0.5]).unwrap();
    let edges = arena.get(root).children.as_ref().unwrap().clone();
    let selected = select_puct(&arena, root, 1.5).unwrap();
    assert_eq!(selected.action_idx, 3);

    // A child value of -0.8 means +0.8 from the parent's perspective.
    let child = edges[1].child;
    arena.get_mut(child).visits = 1;
    arena.get_mut(child).value_sum = -0.8;
    assert_eq!(select_puct(&arena, root, 1.5).unwrap().action_idx, 8);
}

#[test]
fn backup_flips_sign_at_each_ply() {
    let (mut arena, root) = Arena::with_root(1.0);
    expand(&mut arena, root, &[1], &[1.0]).unwrap();
    let child = arena.get(root).children.as_ref().unwrap()[0].child;
    backup(&mut arena, &[root, child], 0.7);
    assert_eq!(arena.get(root).visits, 1);
    assert_eq!(arena.get(child).visits, 1);
    assert!((arena.get(child).value_sum - 0.7).abs() < 1e-12);
    assert!((arena.get(root).value_sum + 0.7).abs() < 1e-12);
}
