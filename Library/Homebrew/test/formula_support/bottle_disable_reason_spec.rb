# typed: false
# frozen_string_literal: true

require "formula_support"

describe BottleDisableReason do
  specify ":unneeded" do
    bottle_disable_reason = described_class.new :unneeded, nil
    expect(bottle_disable_reason).to be_unneeded
    expect(bottle_disable_reason.to_s).to eq("This formula doesn't require compiling.")
  end

  specify ":disabled" do
    bottle_disable_reason = described_class.new :disable, "reason"
    expect(bottle_disable_reason).not_to be_unneeded
    expect(bottle_disable_reason.to_s).to eq("reason")
  end
end
