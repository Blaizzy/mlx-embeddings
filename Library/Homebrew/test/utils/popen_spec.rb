# typed: false
# frozen_string_literal: true

require "utils/popen"

describe Utils do
  describe "::popen_read" do
    it "reads the standard output of a given command" do
      expect(described_class.popen_read("sh", "-c", "echo success").chomp).to eq("success")
      expect($CHILD_STATUS).to be_a_success
    end

    it "can be given a block to manually read from the pipe" do
      expect(
        described_class.popen_read("sh", "-c", "echo success") do |pipe|
          pipe.read.chomp
        end,
      ).to eq("success")
      expect($CHILD_STATUS).to be_a_success
    end

    it "fails when the command does not exist" do
      expect(described_class.popen_read("./nonexistent", err: :out))
        .to eq("brew: command not found: ./nonexistent\n")
      expect($CHILD_STATUS).to be_a_failure
    end
  end

  describe "::popen_write" do
    let(:foo) { mktmpdir/"foo" }

    before { foo.write "Foo\n" }

    it "supports writing to a command's standard input" do
      described_class.popen_write("grep", "-q", "success") do |pipe|
        pipe.write "success\n"
      end
      expect($CHILD_STATUS).to be_a_success
    end

    it "returns the command's standard output before writing" do
      child_stdout = described_class.popen_write("cat", foo, "-") do |pipe|
        pipe.write "Bar\n"
      end
      expect($CHILD_STATUS).to be_a_success
      expect(child_stdout).to eq <<~EOS
        Foo
        Bar
      EOS
    end

    it "returns the command's standard output after writing" do
      child_stdout = described_class.popen_write("cat", "-", foo) do |pipe|
        pipe.write "Bar\n"
      end
      expect($CHILD_STATUS).to be_a_success
      expect(child_stdout).to eq <<~EOS
        Bar
        Foo
      EOS
    end

    it "supports interleaved writing between two reads" do
      child_stdout = described_class.popen_write("cat", foo, "-", foo) do |pipe|
        pipe.write "Bar\n"
      end
      expect($CHILD_STATUS).to be_a_success
      expect(child_stdout).to eq <<~EOS
        Foo
        Bar
        Foo
      EOS
    end
  end
end
