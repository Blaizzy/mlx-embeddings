# typed: false
# frozen_string_literal: true

require "rubocops/platform"

describe RuboCop::Cop::Homebrew::MoveToExtendOS do
  subject(:cop) { described_class.new }

  it "registers an offense when using `OS.linux?`" do
    expect_offense(<<~RUBY)
      OS.linux?
      ^^^^^^^^^ Move `OS.linux?` and `OS.mac?` calls to `extend/os`.
    RUBY
  end

  it "registers an offense when using `OS.mac?`" do
    expect_offense(<<~RUBY)
      OS.mac?
      ^^^^^^^ Move `OS.linux?` and `OS.mac?` calls to `extend/os`.
    RUBY
  end
end
