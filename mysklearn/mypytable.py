import copy
from mysklearn import myutils

# TODO: copy your mypytable.py solution from PA2-PA3 here
class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTabel.
        
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shap MxM (None if empty)
        """

        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        
        Returns:
            tuple: (n_rows, n_cols)
        """

        return (len(self.data), len(self.column_names))

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list
        
        Args:
            col_identifier(str or int): string for a column name or int for a column index
            include_missing_values(bool): True if missing values should be included in the column, False otherwise
            
        Returns:
            list of obj: 1D list of values in the column
        
        Notes:
            Raise ValueError or invalid col_identifier
        """
        col_index = self.__get_column_index(col_identifier)

        if include_missing_values:
            return [row[col_index] for row in self.data]
        else:
            return [row[col_index] for row in self.data if row[col_index] is not None]

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    pass
    
    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        self.data = [row for i, row in enumerate(self.data) if i not in row_indexes_to_drop]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
            
        Returns:
            MyPytable: return self so the caller can write code like: table = MyPyTable().
            
        Notes:
            Use the csv module
            First row of CSV file is assumed to be the header
            Calls convert_to_numeric() after load
        """
        import csv

        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            self.column_names = next(csvreader)
            self.data = list(csvreader)

        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file
        
        Args:
            filename(str): relative path for the CSV file to save the contents to.
            
        Notes:
            Use the csv module
        """
        import csv

        with open(filename, 'w', newline='') as csvfile:
            csvwrite = csv.writer(csvfile)
            csvwrite.writerow(self.column_names)
            csvwrite.writerows(self.data)

    def find_dupicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.
        
        Args:
            key_column_names(list of str): column names to use as row keys
        
        Returns:
            list of list of obj: list of duplicate rows found
            
        Notes:
            Subsequent occurence(s) of a row are considered the duplicate(s). The first instance is not considered a duplicate
        """
        seen = {}
        duplicates = []
        key_column_indexes = [self.__get_column_index(name) for name in key_column_names]

        for row in self.data:
            key = tuple(row[i] for i in key_column_indexes)
            if key in seen:
                duplicates.append(row)
            else:
                seen[key] = True
        
        return duplicates

    def _get_column_index(self, col_identifier):
        """Helper method to get column index from column name or index."""
        if isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError(f"Column '{col_identifier}' not found")
            return self.column_names.index(col_identifier)
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError(f"Column index {col_identifier} out of range")
            return col_identifier
        else:
            raise ValueError("Column identifier must be a string or integer")