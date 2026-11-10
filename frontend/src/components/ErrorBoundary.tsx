"use client";

import { Component, type ReactNode } from "react";

interface Props {
  /** Changing this clears a previous error (e.g. a new session). */
  resetKey: string;
  onError: (message: string) => void;
  children: ReactNode;
}

/** Keeps a failed mesh or texture load from taking the whole studio down. */
export class ErrorBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    this.props.onError(`The 3D proxy could not be displayed: ${error.message}`);
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.failed) this.setState({ failed: false });
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}
