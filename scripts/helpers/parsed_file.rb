require 'fileutils'

class ParsedFile

  def get_latest_file(directory)
    puts "- retrieving latest file in directory: #{directory}"
    Dir.glob("#{directory}/*").max_by(1) {|f| File.mtime(f)}[0]
  end

  def save_to(directory, data)
      # Create directory if does not exist
      FileUtils.mkdir_p directory unless Dir.exists?(directory)

      puts "- Generating datetime stamp"
      #Include time to the filename for uniqueness when fetching multiple times a day
      date_time = Time.new.strftime("%Y-%m-%dT%H_%M_%S")

      # Writing parsed data to file
      puts "- Writing data to file"
      File.write("#{directory}/#{date_time}.txt", data)
  end

end