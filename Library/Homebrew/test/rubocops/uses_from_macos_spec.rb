# frozen_string_literal: true

require "rubocops/uses_from_macos"

describe RuboCop::Cop::FormulaAudit::UsesFromMacos do
  subject(:cop) { described_class.new }

  it "when auditing uses_from_macos dependencies" do
    expect_offense(<<~RUBY)
      class Foo < Formula
        url "https://brew.sh/foo-1.0.tgz"
        homepage "https://brew.sh"

        uses_from_macos "postgresql"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ FormulaAudit/UsesFromMacos: `uses_from_macos` should only be used for macOS dependencies, not postgresql.
      end
    RUBY
  end

  include_examples "formulae exist", described_class::ALLOWED_USES_FROM_MACOS_DEPS
end
