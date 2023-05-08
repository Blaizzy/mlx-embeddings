# frozen_string_literal: true

require "rubocops/rubocop-cask"

describe RuboCop::Cop::Cask::NoDslVersion, :config do
  it "accepts `cask` without a DSL version" do
    expect_no_offenses "cask 'foo' do; end"
  end

  it "reports an offense when `cask` has a DSL version" do
    expect_offense <<~CASK
      cask :v1 => 'foo' do; end
      ^^^^^^^^^^^^^^^^^ Use `cask 'foo'` instead of `cask :v1 => 'foo'`.
    CASK

    expect_correction <<~CASK
      cask 'foo' do; end
    CASK
  end
end
