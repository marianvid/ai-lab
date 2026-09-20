# Source build versions

Keeping an engine's compiled source versions up to date.

Both machines compile llama.cpp from git rather than installing a package, so
this module does what you would otherwise do by hand: report which build is
installed, ask upstream whether there is a newer one, and run the update while
streaming its output to the browser.

On a versioned installation every update is configured and compiled in a new
folder, verified, and only then selected through the stable `current` link.
The previous build remains untouched for rollback. Configuration supplies the
machine-specific CMake flags explicitly; guessing them would risk producing a
working binary that quietly lost an optimisation.

An installation without a configured build root keeps the original behaviour
for backwards compatibility: it checks out the selected tag and rebuilds the
existing `build/` directory in place.

**There are two lines to follow, and which one is a choice.** Upstream tags
almost every commit to master as `b10448` — about seven a day, 7,160 of them in
the checkout when this was measured. On 17 August 2026 they also started a
second line, `v0.1.0` and up, in their own words "stable, slower release
cadence, recommended for downstream distribution and casual users", with
release notes worth reading. The `b` line they call "bleeding edge...
recommended for developers and technical users".

Which line this follows comes from `source.line` in the configuration:
"stable" for `vX.Y.Z`, "nightly" for `b<number>`. Stable is the default,
because an update nobody can read about is not a decision.

The two lines cannot be compared by their numbers — `b10448` and `v0.2.0` are
not on the same scale — so "is there something newer?" is asked of git
instead: is the tag we would move to already in this checkout's history? That
question is exact, it is the same question on both lines, and it stays right
when somebody switches from one to the other.
