# frozen_string_literal: true

shared_examples "parseable arguments" do
  subject(:method_name) do |example|
    example.metadata[:example_group][:parent_example_group][:description]
           .gsub(/^Homebrew\./, "")
  end

  let(:command_name) do
    method_name.gsub(/_args$/, "").tr("_", "-")
  end

  it "can parse arguments" do
    require "dev-cmd/#{command_name}" unless require? "cmd/#{command_name}"

    expect { Homebrew.send(method_name).parse({}) }
      .not_to raise_error
  end
end
