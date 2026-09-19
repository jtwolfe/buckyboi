//! buckyboi: overlay simulation, UX, gaze, and local identity.

pub mod camera;
pub mod draw;
pub mod gaze;
pub mod identity;
pub mod radial;
pub mod sim;
pub mod ux;

pub use gaze::*;
pub use identity::{
    AuthSession, AuthState, EnrollKind, EnrollPhase, EnrollSession, GateMode, GestureAction,
    GestureClass, ProfileStore,
};
pub use radial::*;
pub use sim::*;
pub use ux::*;
