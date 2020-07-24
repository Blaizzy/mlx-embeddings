# frozen_string_literal: true

require "extend/pathname"

describe Pathname, skip: HOMEBREW_PATCHELF_RB.blank? do
  let(:elf_dir) { described_class.new "#{TEST_FIXTURE_DIR}/elf" }
  let(:sho) { elf_dir/"libhello.so.0" }
  let(:exec) { elf_dir/"hello" }

  describe "#interpreter" do
    it "returns interpreter" do
      expect(exec.interpreter).to eq "/lib64/ld-linux-x86-64.so.2"
    end
  end

  describe "#rpath" do
    it "returns nil when absent" do
      expect(exec.rpath).to be_nil
      expect(sho.rpath).to be_nil
    end
  end
end
