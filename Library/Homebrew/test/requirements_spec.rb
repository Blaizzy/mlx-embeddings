# typed: false
# frozen_string_literal: true

require "requirements"

describe Requirements do
  describe "#<<" do
    it "returns itself" do
      expect(subject << Object.new).to be(subject)
    end

    it "merges duplicate requirements" do
      subject << Requirement.new << Requirement.new
      expect(subject.count).to eq(1)
    end
  end
end
