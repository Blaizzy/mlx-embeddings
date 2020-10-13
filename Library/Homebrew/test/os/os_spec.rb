# typed: false
# frozen_string_literal: true

describe OS do
  describe ".kernel_version" do
    it "is not empty" do
      expect(described_class.kernel_version).not_to be_empty
    end
  end
end
