use std::str::from_utf8;
use std::thread;
use std::time::Duration;
use std::{sync::Arc, thread::sleep};

use anyhow::Result;
use crimeline::{
    Order, Relationships, Sharding, Timeline, Uid, Window,
    arena::{Cold, Hot},
};

fn freeze(epoch: u64, duration: u32, entries: &[(Uid, u64, u64, &str)]) -> Result<Arc<Cold>> {
    let mut hot = Hot::new(Window::new(epoch, duration))?;
    for &(uid, cid, ts, text) in entries {
        hot.add(uid, cid, ts, text.as_bytes())?;
    }
    Ok(hot.try_into()?)
}

fn main() -> Result<()> {
    const SEBASTIAN: u32 = 1;

    const ALICE: u32 = 10;
    const BOB: u32 = 20;
    const CAROL: u32 = 30;
    const TROLL: u32 = 99;

    let rels = Arc::new(Relationships::new(Sharding::S2));

    let viewer: Uid = SEBASTIAN;

    for uid in &[ALICE, BOB, CAROL] {
        rels.follows.add(viewer, *uid);
    }

    rels.blocks.add(viewer, TROLL);

    let timeline = Arc::new(Timeline::new(vec![
        freeze(
            1000,
            100,
            &[
                (ALICE, 1, 1010, "gm from alice"),
                (BOB, 2, 1020, "bob's hot take"),
                (TROLL, 3, 1030, "blocked troll"), // should be filtered
                (50, 4, 1040, "stranger waves"),   // not followed
            ],
        )?,
        freeze(
            2000,
            100,
            &[
                (CAROL, 10, 2010, "carol ships code"),
                (BOB, 11, 2050, "bob strikes again"),
            ],
        )?,
    ]));

    println!("initial timeline ({} arenas)", timeline.len());

    let mut handles = Vec::new();

    for reader_id in 0..3 {
        let tl = Arc::clone(&timeline);
        let r = Arc::clone(&rels);

        handles.push(thread::spawn(move || {
            sleep(Duration::from_millis(reader_id * 30));

            let mut iter = tl.iter(0, Order::Asc);
            let mut count = 0;

            while let Some(e) = iter.next() {
                if !r.follows.contains(viewer, e.uid) {
                    continue;
                }

                if r.blocks.contains(viewer, e.uid) {
                    continue;
                }

                count += 1;

                let (_cid, blob) = e.resolve();
                let text = from_utf8(blob).unwrap();

                println!("  ts={} uid={:>2}  {text}", e.timestamp(), e.uid);
            }

            println!("[reader {reader_id}] saw {count} entries\n");
        }));
    }

    {
        let tl = Arc::clone(&timeline);

        handles.push(thread::spawn(move || {
            sleep(Duration::from_millis(10));

            let arena = freeze(
                3000,
                100,
                &[
                    (ALICE, 20, 3010, "alice late post"),
                    (CAROL, 21, 3020, "carol too"),
                ],
            )
            .expect("freeze");

            tl.add(arena);
            println!(
                "[producer] added arena at epoch 3000 (now {} arenas)\n",
                tl.len()
            );
        }));
    }

    {
        let tl = Arc::clone(&timeline);
        handles.push(thread::spawn(move || {
            sleep(Duration::from_millis(50));
            tl.remove(1000);
            println!(
                "[janitor] removed arena at epoch 1000 (now {} arenas)\n",
                tl.len()
            );
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    println!("final timeline ({} arenas)", timeline.len());

    let mut iter = timeline.iter(0, Order::Desc);

    while let Some(e) = iter.next() {
        if !rels.follows.contains(viewer, e.uid) {
            continue;
        }

        if rels.blocks.contains(viewer, e.uid) {
            continue;
        }

        let (_cid, blob) = e.resolve();
        let text = from_utf8(blob).unwrap_or("<binary>");

        println!("  ts={} uid={:>2}  {text}", e.timestamp(), e.uid);
    }

    Ok(())
}
