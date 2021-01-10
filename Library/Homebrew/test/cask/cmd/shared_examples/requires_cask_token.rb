# typed: false
# frozen_string_literal: true

shared_examples "a command that requires a Cask token" do
  context "when no Cask is specified" do
    it "raises an exception " do
      expect {
        described_class.run
      }.to raise_error(UsageError, /this command requires a .*cask.* argument/)
    end
  end
end
