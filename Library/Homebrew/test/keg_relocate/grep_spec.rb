# typed: false
# frozen_string_literal: true

require "keg_relocate"

describe Keg do
  subject(:keg) { described_class.new(HOMEBREW_CELLAR/"foo/1.0.0") }

  let(:dir) { HOMEBREW_CELLAR/"foo/1.0.0" }
  let(:text_file) { dir/"file.txt" }
  let(:binary_file) { dir/"file.bin" }

  before do
    dir.mkpath
  end

  def setup_text_file
    text_file.atomic_write <<~EOS
      #{dir}/file.txt
      /foo#{dir}/file.txt
      foo/bar:#{dir}/file.txt
      foo/bar:/foo#{dir}/file.txt
      #{dir}/bar.txt:#{dir}/baz.txt
    EOS
  end

  def setup_binary_file
    binary_file.atomic_write <<~EOS
      \x00
    EOS
  end

  describe "#each_unique_file_matching" do
    specify "find string matches to path" do
      setup_text_file

      string_matches = Set.new
      keg.each_unique_file_matching(dir) do |file|
        string_matches << file
      end

      expect(string_matches.size).to eq 1
    end
  end

  describe "#each_unique_binary_file" do
    specify "find null bytes in binaries" do
      setup_binary_file

      binary_matches = Set.new
      keg.each_unique_binary_file do |file|
        binary_matches << file
      end

      expect(binary_matches.size).to eq 1
    end
  end
end
