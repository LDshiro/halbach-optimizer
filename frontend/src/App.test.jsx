import { render, screen, waitFor } from "@testing-library/react";

import App from "./App";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plotly-mock" />,
}));

test("renders read-only GUI shell", async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ runs: [{ path: "runs/demo", label: "runs/demo" }] }),
  });

  render(<App />);

  expect(screen.getByText("Halbach Read-only GUI")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Refresh Read-only Data" })).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getAllByDisplayValue("runs/demo")).toHaveLength(2);
  });
});
