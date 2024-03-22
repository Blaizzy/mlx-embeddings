# frozen_string_literal: true

RSpec.shared_examples "parseable arguments" do |argv: nil|
  subject(:method_name) { "#{command_name.tr("-", "_")}_args" }

  let(:command_name) do |example|
    example.metadata[:example_group][:parent_example_group][:description].delete_prefix("brew ")
  end

  it "can parse arguments" do
    if described_class
      argv ||= described_class.parser.instance_variable_get(:@min_named_args)&.times&.map { "argument" }
      argv ||= []
      cmd = described_class.new(argv)
      expect(cmd.args).to be_a Homebrew::CLI::Args
    else
      require "dev-cmd/#{command_name}" unless require? "cmd/#{command_name}"
      parser = Homebrew.public_send(method_name)
      expect(parser).to respond_to(:parse)
    end
  end
end
