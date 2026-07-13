import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("DorkChat streams a cited response and restores bounded browser history", async ({
  page,
}) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        ready: true,
        version: "0.2.0",
        model_loaded: true,
        rag_chunks: 24,
        active_provider: "local_gpt",
        model: { name: "DorkLLM-SFT", degraded: false },
      }),
    });
  });

  await page.route("**/api/v1/chat/stream", async (route) => {
    const payload = route.request().postDataJSON();
    expect(payload.message).toBeUndefined();
    expect(payload.history).toBeUndefined();
    expect(payload.messages.at(-1)).toEqual({
      role: "user",
      content: "How is this answer grounded?",
    });
    expect(
      payload.messages.filter(
        ({ content }) => content === "How is this answer grounded?",
      ).length,
    ).toBe(1);
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "cache-control": "no-cache" },
      body: [
        'event: meta\ndata: {"request_id":"e2e-1","model":"DorkLLM-SFT","active_provider":"local_gpt"}\n\n',
        'event: delta\ndata: {"delta":"The answer is grounded "}\n\n',
        'event: delta\ndata: {"delta":"in retrieved evidence [1]."}\n\n',
        'event: citation\ndata: {"citation":{"marker":1,"source":"docs/rag_design.md","snippet":"Retrieved chunks retain source provenance.","score":0.94}}\n\n',
        'event: done\ndata: {"mode":"rag","model":"DorkLLM-SFT","latency_ms":48}\n\n',
      ].join(""),
    });
  });

  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();

  await expect(
    page.getByRole("heading", {
      name: "Ask the system. Inspect the evidence.",
    }),
  ).toBeVisible();
  await expect(page.getByText("DorkLLM-SFT", { exact: true })).toBeVisible();
  await expect(page.getByText("local_gpt", { exact: true })).toBeVisible();

  await page
    .getByLabel("Message DorkChat")
    .fill("How is this answer grounded?");
  await page.getByRole("button", { name: "Send" }).click();

  const response = page.getByLabel("DorkLLM response");
  await expect(response).toContainText(
    "The answer is grounded in retrieved evidence [1].",
  );
  await expect(response).toContainText("docs/rag_design.md");
  await expect(response).toContainText(
    "Retrieved chunks retain source provenance.",
  );
  await expect(page.getByText("48 ms", { exact: true })).toBeVisible();

  const accessibility = await new AxeBuilder({ page })
    .include("main")
    .analyze();
  expect(accessibility.violations).toEqual([]);

  await page.reload();
  await expect(page.getByLabel("DorkLLM response")).toContainText(
    "The answer is grounded in retrieved evidence [1].",
  );
  await expect(page.getByText("Move from claim to evidence.")).toBeHidden();
});
