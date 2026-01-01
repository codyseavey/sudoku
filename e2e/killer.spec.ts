import { test, expect } from '@playwright/test';

test('Killer Sudoku loads and renders correctly', async ({ page }) => {
  // Go to home page
  await page.goto('http://localhost:8080');

  // Select Killer Sudoku
  await page.selectOption('select.difficulty-select', 'killer');

  // Click New Game
  await page.click('button:has-text("Start Game")');

  // Check if grid is visible
  const grid = page.locator('.grid');
  await expect(grid).toBeVisible();

  // Check if cages are rendered (by checking for cage-sum or cage classes)
  // We use the presence of 'cage-sum' elements
  const cageSum = page.locator('.cage-sum').first();
  await expect(cageSum).toBeVisible();

  // Check if the board has empty cells (Killer Hard default behavior if difficulty logic wasn't fully hooked up,
  // but wait, we are using default difficulty which might be Easy or Medium)
  // The LandingPage.vue (which we haven't seen but assume exists) likely defaults difficulty.

  // Let's verify that we have some inputs
  const inputs = page.locator('.value-input');
  const count = await inputs.count();
  expect(count).toBe(81);
});
