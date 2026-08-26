import React from "react";
import { Alert, Button } from "antd";

interface Props {
  /** Human-readable label shown in the fallback (e.g. "分析面板"). */
  label?: string;
  children: React.ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches render errors in child components and shows a non-fatal fallback
 * instead of white-screening the entire app.
 *
 * Usage:
 *   <PanelErrorBoundary label="因子归因">
 *     <FactorAttributionPanel ... />
 *   </PanelErrorBoundary>
 */
export class PanelErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error(`[ErrorBoundary:${this.props.label ?? "panel"}]`, error, info.componentStack);
  }

  private handleReset = (): void => {
    this.setState({ error: null });
  };

  render(): React.ReactNode {
    if (this.state.error === null) {
      return this.props.children;
    }

    const label = this.props.label ?? "此面板";

    return (
      <Alert
        type="error"
        showIcon
        style={{ margin: 16 }}
        message={`${label}渲染异常`}
        description={
          <>
            <div style={{ marginBottom: 8, fontFamily: "monospace", fontSize: 12, wordBreak: "break-all" }}>
              {this.state.error.message}
            </div>
            <Button size="small" onClick={this.handleReset}>
              重试
            </Button>
          </>
        }
      />
    );
  }
}
